"""
Command-Line Interface (CLI) for Text Generation:
Runs autoregressive generation using the preallocated StaticKVCache.
Supports custom prompts, temperature scaling, top-p (nucleus) sampling,
and checkpoint loading.
"""

import argparse
import os
import torch

from modern_decoder_llm.model import DecoderLLM
from modern_decoder_llm.sampler import generate_with_cache
from modern_decoder_llm.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="Autoregressive Text Generation with Static KV Cache")
    parser.add_argument("--prompt", type=str, default="Once upon a time, there was a little bird", help="Initial prompt text")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json", help="Path to tokenizer JSON")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0.0 = greedy)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) cumulative probability")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("MODERN DECODER LLM: TEXT GENERATION")
    print(f"Device: {device} | Temperature: {args.temperature} | Top-p: {args.top_p}")
    print("=" * 70)

    # 1. Load or initialize tokenizer
    if os.path.exists(args.tokenizer):
        print(f"Loading tokenizer from {args.tokenizer}...")
        tokenizer = Tokenizer.load(args.tokenizer)
    else:
        print(f"Tokenizer not found at {args.tokenizer}. Using base byte-level tokenizer.")
        tokenizer = Tokenizer(vocab_size=4096)

    # 2. Instantiate model architecture (5.57M tied parameters)
    model = DecoderLLM(
        vocab_size=4096,
        max_seq_len=256,
        hidden_dim=256,
        depth=6,
        n_heads=8,
        n_kv_heads=2,
        head_dim=32,
        ffn_dim=768,
        rope_theta=10000.0,
        tie_weights=True,
        norm_eps=1e-6,
    ).to(device)

    # 3. Load checkpoint weights if present
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint weights from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(state_dict)
    else:
        print(f"No checkpoint found at {args.checkpoint}. Running inference with initialized model.")

    model.eval()

    # 4. Tokenize prompt
    prompt_ids = tokenizer.encode(args.prompt)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    print(f"\nPrompt: \"{args.prompt}\" ({len(prompt_ids)} tokens)")
    print("Generating tokens with preallocated StaticKVCache...")

    # 5. Generate with static KV cache
    output_tokens = generate_with_cache(
        model=model,
        prompt_tokens=prompt_tensor,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_id,
    )

    generated_ids = output_tokens[0].tolist()
    decoded_text = tokenizer.decode(generated_ids)

    print("-" * 70)
    print("Generated Text:")
    print(decoded_text)
    print("-" * 70)
    print(f"Total Sequence Length: {len(generated_ids)} tokens ({len(generated_ids) - len(prompt_ids)} new tokens generated)\n")


if __name__ == "__main__":
    main()

