"""Collect responses whose final answer is wrong from gen_data/generate.py (or eval/eval.py) outputs.

Each wrong response is written as one record with the reason it failed:
  truncated     cut off at max_tokens (unfinished <think> block, or finish_reason == "length")
  no_answer     finished but no final answer could be extracted
  wrong_answer  extracted answer does not match the gold answer
Output mirrors the input layout: <output_dir>/<org>/<model>/<data>/<prompt>/<cfg>.jsonl

Example:
    python stats/collect_wrong.py \
        --input_files outputs/gen_data/Qwen/Qwen3-0.6B/math300_train/prompt_simple/t0.7_k4_s0_e300.jsonl
"""
import argparse
import json
import os
import sys
from collections import Counter

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "..", "gen_data"))
from convert_to_sft import is_complete, load_jsonl  # noqa: E402

REASONS = ["truncated", "no_answer", "wrong_answer"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", type=str, nargs="+", required=True, help="generation jsonl files")
    parser.add_argument("--output_dir", type=str, default="outputs/stats/wrong")
    parser.add_argument("--reasons", type=lambda s: s.split(","), default=REASONS,
                        help=f"comma-separated subset of {','.join(REASONS)} to keep (default: all)")
    return parser.parse_args()


def failure_reason(response, answer, finish_reason=None):
    if finish_reason == "length" or not is_complete(response):
        return "truncated"
    if not (answer and answer.strip()):
        return "no_answer"
    return "wrong_answer"


def output_path(path, output_dir):
    """Keep the last 5 path parts (<org>/<model>/<data>/<prompt>/<cfg>.jsonl) under output_dir."""
    parts = os.path.normpath(path).split(os.sep)
    return os.path.join(output_dir, *parts[-5:])


def main():
    args = parse_args()
    bad = set(args.reasons) - set(REASONS)
    if bad:
        raise ValueError(f"unknown reasons {bad}; choose from {REASONS}")

    rows = []
    for path in args.input_files:
        records = load_jsonl(path)
        if records and "answers_correctness" not in records[0]:
            raise ValueError(f"{path} has no answers_correctness; generate without --no_check")
        wrong, counts, questions = [], Counter(), set()
        n_responses = 0
        for rec in records:
            finish = rec.get("finish_reasons", [None] * len(rec["generated_responses"]))
            for j, (resp, ans, ok, fr) in enumerate(zip(rec["generated_responses"], rec["generated_answers"],
                                                        rec["answers_correctness"], finish)):
                n_responses += 1
                if ok:
                    continue
                reason = failure_reason(resp, ans, fr)
                counts[reason] += 1
                if reason not in args.reasons:
                    continue
                questions.add(rec["idx"])
                wrong.append({
                    "idx": rec["idx"],
                    "response_idx": j,
                    "reason": reason,
                    "question": rec["question"],
                    "gold_answer": rec["gold_answer"],
                    "generated_answer": ans,
                    "num_correct_for_question": sum(map(bool, rec["answers_correctness"])),
                    "response_chars": len(resp),
                    "response": resp,
                })

        out_file = output_path(path, args.output_dir)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            for w in wrong:
                f.write(json.dumps(w, ensure_ascii=False) + "\n")
        parts = os.path.normpath(path).split(os.sep)
        rows.append([f"{parts[-5]}/{parts[-4]}" if len(parts) >= 5 else path, n_responses, sum(counts.values()),
                     *[counts[r] for r in REASONS], len(questions), out_file])

    header = ["model", "responses", "wrong", *REASONS, "questions", "saved to"]
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    fmt = lambda row: "  ".join(str(v).ljust(w) if i in (0, len(row) - 1) else str(v).rjust(w)
                                for i, (v, w) in enumerate(zip(row, widths)))
    print(fmt(header))
    print("-" * len(fmt(header)))
    for row in rows:
        print(fmt(row))
    print(f"\nkept reasons: {','.join(args.reasons)}  |  questions = questions with >=1 kept wrong response")


if __name__ == "__main__":
    main()
