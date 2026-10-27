"""Convert answer-guided generations (gen_data/generate.py output) into an SFT dataset for LLaMA-Factory.

Which responses are kept is set by --select:
  correct (default) correct final answer only, no other filtering
  clean   correct, complete (not cut off at max_tokens), and not leaking the hint
          (e.g. "which matches the given answer"), since the answer is not in the prompt at training time
  all     every response
At most --max_per_question kept responses (default 1) are sampled at random per question; a question with no
kept response is skipped. Records are written in Alpaca format (instruction / input / output, same as SLM_REASONING/data) and the
dataset is registered in <data_dir>/dataset_info.json.

Example:
    python gen_data/convert_to_sft.py \
        --input_file outputs/gen_data/Qwen/Qwen3-0.6B/math10_train/t0.7_k4_s0_e10.jsonl \
        --dataset_name math10_qwen3_0.6b_guided
"""
import argparse
import json
import os
import random
import re
from collections import Counter

# same instruction as the reference SFT sets and the eval prompt (SLM_REASONING qwen-instruct)
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

# phrasings that only make sense if the model was shown the answer
LEAK_PATTERNS = [
    r"\b(given|provided|stated|expected|target|known)\s+(correct\s+)?(final\s+)?answer",
    r"\bcorrect\s+final\s+answer\b",
    r"\banswer\s+(is|was)\s+supposed\s+to\s+be\b",
    r"\b(problem|question|prompt|statement)\s+(says|said|states|stated|tells|told|gives|gave|mentions|mentioned|claims)"
    r"\s+(that\s+)?(the\s+)?(correct\s+|final\s+)*answer",
    r"\bgiven\s+(that\s+)?the\s+(correct\s+|final\s+)*answer\s+is\b",
    r"\baccording\s+to\s+the\s+(problem|question|prompt),\s+the\s+(correct\s+|final\s+)*answer",
    # prompt.py calls the answer a "private answer key"; "key point" / "the key here" are not leaks
    r"\b(answer|private)\s+key\b",
    r"\bkey(['’]s)?\s+answer\b",
    r"\bthe\s+key\s+(says|said|is|was|given|provided|suggests?|confirms?)\b",
    r"\b(matches|match|matching|contradicts|confirms|is)\s+the\s+key\b",
    r"\b(given|provided|stated)\s+in\s+the\s+key\b",
]
LEAK_RE = re.compile("|".join(LEAK_PATTERNS), re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="jsonl written by gen_data/generate.py")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="key in dataset_info.json and output file stem (default: derived from input path)")
    parser.add_argument("--max_per_question", type=int, default=1,
                        help="max kept responses per question, sampled at random (-1: keep all)")
    parser.add_argument("--seed", type=int, default=0, help="seed for sampling responses per question")
    parser.add_argument("--select", type=str, default="correct", choices=["correct", "clean", "all"],
                        help="correct: correct only; clean: correct + complete + no hint leak; all: every response")
    parser.add_argument("--strip_think", action="store_true", help="drop the <think>...</think> block, keep the final solution")
    parser.add_argument("--instruction", type=str, default=INSTRUCTION)
    return parser.parse_args()


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_complete(response):
    # an unfinished thinking block means the generation hit max_tokens
    return "<think>" not in response or "</think>" in response


def strip_think(response):
    return response.split("</think>", 1)[1].strip() if "</think>" in response else response.strip()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    records = load_jsonl(args.input_file)
    if args.select != "all" and records and "answers_correctness" not in records[0]:
        raise ValueError(f"{args.input_file} has no answers_correctness; rerun generate.py without --no_check")

    if args.dataset_name is None:
        # outputs/gen_data/<org>/<model>/<data>/<prompt_type>/<cfg>.jsonl -> <data>_<model>_<prompt_type>_<cfg>
        parts = os.path.normpath(args.input_file).split(os.sep)
        model, data, prompt_type, cfg = parts[-4], parts[-3], parts[-2], os.path.splitext(parts[-1])[0]
        args.dataset_name = re.sub(r"[^A-Za-z0-9_.]+", "_", f"{data}_{model}_{prompt_type}_{cfg}_{args.select}")

    stats = Counter()
    out, n_questions_kept = [], 0
    for rec in records:
        kept = []
        correctness = rec.get("answers_correctness", [None] * len(rec["generated_responses"]))
        for response, correct in zip(rec["generated_responses"], correctness):
            stats["responses"] += 1
            if args.select == "clean" and not is_complete(response):
                stats["drop_truncated"] += 1
            elif args.select in ("clean", "correct") and not correct:
                stats["drop_incorrect"] += 1
            elif args.select == "clean" and LEAK_RE.search(response):
                stats["drop_leaky"] += 1
            else:
                kept.append(strip_think(response) if args.strip_think else response.strip())
        if 0 < args.max_per_question < len(kept):
            kept = rng.sample(kept, args.max_per_question)
        n_questions_kept += bool(kept)
        out += [{"instruction": args.instruction, "input": rec["question"], "output": r} for r in kept]

    os.makedirs(args.data_dir, exist_ok=True)
    file_name = f"{args.dataset_name}.json"
    with open(os.path.join(args.data_dir, file_name), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    info_path = os.path.join(args.data_dir, "dataset_info.json")
    info = {}
    if os.path.exists(info_path):
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
    info[args.dataset_name] = {"file_name": file_name}
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"select={args.select} | responses: {stats['responses']}, dropped incorrect: {stats['drop_incorrect']}, "
          f"truncated: {stats['drop_truncated']}, leaky: {stats['drop_leaky']}")
    print(f"kept {len(out)} responses covering {n_questions_kept}/{len(records)} questions")
    print(f"wrote {os.path.join(args.data_dir, file_name)}, registered '{args.dataset_name}' in {info_path}")


if __name__ == "__main__":
    main()
