"""
Llama-3-Style Decoder Pretraining Pipeline on TinyStories:
1. Optimizer: AdamW with weight decay restricted strictly to 2D linear weight matrices.
   (Norm scale parameters, biases, and token embeddings have 0.0 weight decay).
2. Schedule: Linear warmup followed by Cosine learning rate decay down to min_lr.
3. Stability: Gradient clipping (L2 norm <= 1.0) and gradient accumulation.
4. Validation: Periodic evaluation of Cross-Entropy Loss and Perplexity (exp(loss)).
5. Checkpointing: Saves best and latest model checkpoints.
"""

import math
import os
from typing import Optional, Tuple
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from modern_decoder_llm.model import DecoderLLM
from modern_decoder_llm.dataset import MemmapTokenDataset, prepare_tinystories_data


def configure_optimizers(
    model: DecoderLLM,
    weight_decay: float,
    learning_rate: float,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """
    Partitions model parameters into decay and no-decay groups.
    
    Rule:
    - Weight decay is applied ONLY to 2D projection matrices (Linear layers).
    - RMSNorm gamma parameters (1D) and Token Embeddings (table) receive 0.0 weight decay.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Embeddings and output LM head (tied) are exempted from weight decay
        if "tok_embeddings" in name or "lm_head" in name:
            no_decay_params.append(param)
        elif param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    num_decay = sum(p.numel() for p in decay_params)
    num_nodecay = sum(p.numel() for p in no_decay_params)
    print(
        f"Optimizer Parameter Groups: "
        f"{len(decay_params)} decay tensors ({num_decay:,} params), "
        f"{len(no_decay_params)} no-decay tensors ({num_nodecay:,} params)"
    )

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=betas,
        eps=eps,
    )
    return optimizer


def get_cosine_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """
    Computes current learning rate using linear warmup and cosine decay:
    1. Linear warmup for step < warmup_steps: lr = max_lr * (step + 1) / warmup_steps
    2. Cosine decay for warmup_steps <= step <= max_steps
    3. Flat min_lr for step > max_steps
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def evaluate_validation(
    model: DecoderLLM,
    val_loader: DataLoader,
    device: torch.device,
    eval_steps: int = 50,
) -> Tuple[float, float]:
    """
    Computes average validation loss and perplexity:
        Perplexity = exp(Loss)
    """
    model.eval()
    total_loss = 0.0
    steps_evaluated = 0

    for step, (x, y) in enumerate(val_loader):
        if step >= eval_steps:
            break
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))
        total_loss += loss.item()
        steps_evaluated += 1

    avg_loss = total_loss / max(1, steps_evaluated)
    # Clamp perplexity to prevent math overflow if loss is large
    perplexity = math.exp(min(avg_loss, 20.0))
    model.train()
    return avg_loss, perplexity


def train(
    config_path: str = "configs/tinystories.yaml",
    max_steps_override: Optional[int] = None,
) -> DecoderLLM:
    """Main pretraining loop."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Set random seed
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Build or verify dataset
    train_bin, val_bin, tokenizer = prepare_tinystories_data()

    seq_len = cfg["model"]["max_seq_len"]
    batch_size = cfg["training"]["batch_size"]
    grad_accum_steps = cfg["training"]["grad_accum_steps"]
    max_steps = max_steps_override or cfg["training"]["max_steps"]
    warmup_steps = cfg["training"]["warmup_steps"]
    max_lr = float(cfg["training"]["lr"])
    min_lr = float(cfg["training"]["min_lr"])
    weight_decay = float(cfg["training"]["weight_decay"])
    grad_clip = float(cfg["training"]["grad_clip"])
    eval_interval = cfg["training"]["eval_interval"]
    eval_steps = cfg["training"]["eval_steps"]

    train_dataset = MemmapTokenDataset(bin_path=train_bin, seq_len=seq_len)
    val_dataset = MemmapTokenDataset(bin_path=val_bin, seq_len=seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Instantiate DecoderLLM
    model = DecoderLLM(
        vocab_size=cfg["model"]["vocab_size"],
        max_seq_len=cfg["model"]["max_seq_len"],
        hidden_dim=cfg["model"]["hidden_dim"],
        depth=cfg["model"]["depth"],
        n_heads=cfg["model"]["n_heads"],
        n_kv_heads=cfg["model"]["n_kv_heads"],
        head_dim=cfg["model"]["head_dim"],
        ffn_dim=cfg["model"]["ffn_dim"],
        rope_theta=cfg["model"]["rope_theta"],
        tie_weights=cfg["model"]["tie_weights"],
        norm_eps=float(cfg["model"]["norm_eps"]),
    ).to(device)

    optimizer = configure_optimizers(model, weight_decay, max_lr)

    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float("inf")

    model.train()
    optimizer.zero_grad()

    data_iter = iter(train_loader)
    progress_bar = tqdm(range(max_steps), desc="Pretraining")

    for step in progress_bar:
        # Cosine learning rate scheduling
        current_lr = get_cosine_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        accumulated_loss = 0.0

        # Gradient accumulation micro-steps
        for micro_step in range(grad_accum_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            x, y = x.to(device), y.to(device)

            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))
            # Scale loss for gradient accumulation
            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

            accumulated_loss += loss.item() / grad_accum_steps

        # Clip global gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()
        optimizer.zero_grad()

        progress_bar.set_postfix({
            "loss": f"{accumulated_loss:.4f}",
            "lr": f"{current_lr:.2e}",
            "gnorm": f"{grad_norm:.2f}",
        })

        # Periodic validation evaluation
        if (step + 1) % eval_interval == 0 or (step + 1) == max_steps:
            val_loss, perplexity = evaluate_validation(model, val_loader, device, eval_steps)
            print(
                f"\n[Step {step + 1}/{max_steps}] "
                f"Train Loss: {accumulated_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val PPL: {perplexity:.2f}"
            )

            # Checkpoint saving
            latest_path = "checkpoints/latest_model.pt"
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
            }, latest_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = "checkpoints/best_model.pt"
                torch.save({
                    "step": step + 1,
                    "model_state": model.state_dict(),
                    "val_loss": val_loss,
                }, best_path)
                print(f"--> Saved new best model checkpoint to {best_path}")

    print("\nPretraining complete!")
    return model


if __name__ == "__main__":
    train()

