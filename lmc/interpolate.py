"""Linearly interpolate between two checkpoints of the same architecture, for linear mode connectivity (LMC).

For each alpha the weights are theta(alpha) = (1 - alpha) * theta_start + alpha * theta_end, computed and saved in
fp32 by default: fine-tuning moves most weights only 0-2 bf16 steps, so saving in bf16 would round every weight to one of
the two endpoints and the real position on the line would not be alpha. config.json gets the same dtype, so vLLM
(dtype=auto) also runs the model in it. alpha = 0 is the start model (e.g. the base model), alpha = 1 the end model (e.g. a
fine-tuned checkpoint). Config, tokenizer and chat template are copied from the end model, so every interpolated
model loads and evaluates like it. Tied lm_head weights are skipped when only one side stores them.
Output: <output_dir>/alpha-<alpha>/

Example:
    python lmc/interpolate.py \
        --start_model Qwen/Qwen3-0.6B \
        --end_model /helios-storage/helios4-data/cuong/model/RSI/Qwen3-0.6B_math12k_prompt_simple_correct/checkpoint-28565 \
        --output_dir /helios-storage/helios4-data/cuong/model/RSI/Qwen3-0.6B_math12k_prompt_simple_correct_lmc_checkpoint-28565 \
        --alphas 0.2,0.4,0.6,0.8
"""
import argparse
import glob
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# files copied from the end model next to the interpolated weights (training state is left out)
COPY_FILES = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
              "chat_template.jinja", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"]
TIED_KEYS = {"lm_head.weight"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_model", type=str, required=True, help="local dir or hub id (alpha = 0)")
    parser.add_argument("--end_model", type=str, required=True, help="local dir or hub id (alpha = 1)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--alphas", type=lambda s: [float(a) for a in s.split(",")], default=[0.2, 0.4, 0.6, 0.8],
                        help="comma-separated interpolation weights of the end model")
    parser.add_argument("--save_dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"],
                        help="dtype of the saved weights and of config.json")
    return parser.parse_args()


def resolve(path):
    if os.path.isdir(path):
        return path
    from huggingface_hub import snapshot_download
    return snapshot_download(path)


def load_state_dict(model_dir):
    state = {}
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no *.safetensors in {model_dir}")
    for file in files:
        with safe_open(file, "pt") as f:
            state.update({k: f.get_tensor(k) for k in f.keys()})
    return state


def main():
    args = parse_args()
    save_dtype = getattr(torch, args.save_dtype)
    start_dir, end_dir = resolve(args.start_model), resolve(args.end_model)
    start, end = load_state_dict(start_dir), load_state_dict(end_dir)

    with open(os.path.join(end_dir, "config.json")) as f:
        tied = json.load(f).get("tie_word_embeddings", False)
    skip = TIED_KEYS if tied else set()
    only_start, only_end = set(start) - set(end) - skip, set(end) - set(start) - skip
    if only_start or only_end:
        raise ValueError(f"parameter names differ: only in start {sorted(only_start)}, only in end {sorted(only_end)}")
    keys = [k for k in end if k in start]
    for k in keys:
        if start[k].shape != end[k].shape:
            raise ValueError(f"shape mismatch for {k}: {tuple(start[k].shape)} vs {tuple(end[k].shape)}")

    # distance between the two endpoints, for reference
    sq = sum((end[k].float() - start[k].float()).pow(2).sum().item() for k in keys)
    norm = sum(start[k].float().pow(2).sum().item() for k in keys)
    print(f"start: {start_dir}\nend:   {end_dir}\n|end - start| = {sq ** 0.5:.4f} "
          f"({100 * (sq / norm) ** 0.5:.3f}% of |start|) over {len(keys)} tensors")

    for alpha in args.alphas:
        out_dir = os.path.join(args.output_dir, f"alpha-{alpha:g}")
        os.makedirs(out_dir, exist_ok=True)
        merged = {k: ((1 - alpha) * start[k].float() + alpha * end[k].float()).to(save_dtype).contiguous()
                  for k in keys}
        save_file(merged, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
        for name in COPY_FILES:
            if os.path.exists(os.path.join(end_dir, name)):
                shutil.copy(os.path.join(end_dir, name), out_dir)
        with open(os.path.join(out_dir, "config.json")) as f:
            config = json.load(f)
        config["dtype"] = args.save_dtype
        config.pop("torch_dtype", None)
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        with open(os.path.join(out_dir, "interpolation.json"), "w") as f:
            json.dump({"start_model": args.start_model, "end_model": args.end_model, "alpha": alpha,
                       "formula": "(1 - alpha) * start + alpha * end", "dtype": args.save_dtype}, f, indent=2)
        # actual position on the line after casting to save_dtype (equals alpha unless rounding moved it)
        num = sum(((merged[k].float() - start[k].float()) * (end[k].float() - start[k].float())).sum().item() for k in keys)
        print(f"alpha={alpha:g} (effective {num / sq:.4f}) -> {out_dir}")


if __name__ == "__main__":
    main()
