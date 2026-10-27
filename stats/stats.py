"""Statistics of model responses written by gen_data/generate.py (or eval/eval.py).

For each input jsonl it reports accuracy, per-question solve rates, how many responses were cut off,
had no answer or leaked the hint (same rules as gen_data/convert_to_sft.py), and response lengths
(characters, optionally tokens), split into the <think> part and the final solution. Several files are
compared side by side, and the results are saved as json and csv.

Example:
    python stats/stats.py \
        --input_files outputs/gen_data/Qwen/Qwen3-0.6B/math300_train/prompt/t0.7_k4_s0_e300.jsonl \
                      outputs/gen_data/Qwen/Qwen3-4B/math300_train/prompt/t0.7_k4_s0_e300.jsonl \
        --count_tokens
"""
import argparse
import csv
import json
import os
import statistics
import sys
from collections import Counter

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "..", "gen_data"))
from convert_to_sft import LEAK_RE, is_complete, load_jsonl  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", type=str, nargs="+", required=True, help="generation jsonl files")
    parser.add_argument("--output_dir", type=str, default="outputs/stats")
    parser.add_argument("--output_name", type=str, default=None, help="file stem for json/csv (default: from inputs)")
    parser.add_argument("--count_tokens", action="store_true",
                        help="also count tokens with the model's tokenizer (taken from the input path)")
    parser.add_argument("--tokenizer", type=str, default=None, help="tokenizer to use for every file (overrides path)")
    return parser.parse_args()


def describe_path(path):
    """outputs/<task>/<org>/<model>/<data>/<prompt>/<cfg>.jsonl -> fields (best effort for other layouts)."""
    parts = os.path.normpath(path).split(os.sep)
    cfg = os.path.splitext(parts[-1])[0]
    if len(parts) >= 5:
        return {"model": f"{parts[-5]}/{parts[-4]}", "data": parts[-3], "prompt": parts[-2], "cfg": cfg}
    return {"model": "?", "data": "?", "prompt": "?", "cfg": cfg}


def split_think(response):
    if "</think>" in response:
        think, final = response.split("</think>", 1)
        return think.replace("<think>", "", 1), final
    if "<think>" in response:  # cut off while thinking
        return response.replace("<think>", "", 1), ""
    return "", response


def summary(values):
    if not values:
        return {"mean": None, "median": None, "p90": None, "max": None}
    values = sorted(values)
    return {"mean": round(statistics.mean(values), 1), "median": statistics.median(values),
            "p90": values[min(len(values) - 1, int(0.9 * len(values)))], "max": values[-1]}


def file_stats(path, tokenizer=None):
    records = load_jsonl(path)
    n_q = len(records)
    responses = [r for rec in records for r in rec["generated_responses"]]
    correctness = [bool(c) for rec in records for c in rec.get("answers_correctness", [])]
    answers = [a for rec in records for a in rec.get("generated_answers", [])]
    has_labels = len(correctness) == len(responses)
    n_r = len(responses)

    complete = [is_complete(r) for r in responses]
    leaky = [bool(LEAK_RE.search(r)) for r in responses]
    no_answer = [not (a and a.strip()) for a in answers] if answers else [False] * n_r
    clean = [c and comp and not lk for c, comp, lk in zip(correctness, complete, leaky)] if has_labels else []

    stats = {**describe_path(path), "file": path, "questions": n_q, "responses": n_r,
             "samples_per_question": round(n_r / max(n_q, 1), 2)}
    if has_labels:
        n_correct = Counter(sum(map(bool, rec["answers_correctness"])) for rec in records)
        k = max(len(rec["answers_correctness"]) for rec in records)
        stats.update({
            "accuracy": round(sum(correctness) / max(n_r, 1), 4),
            "solved_any": round(sum(v for c, v in n_correct.items() if c > 0) / max(n_q, 1), 4),
            "solved_all": round(sum(v for c, v in n_correct.items() if c == k) / max(n_q, 1), 4),
            "solved_none": round(n_correct.get(0, 0) / max(n_q, 1), 4),
            "correct_per_question_hist": {str(c): n_correct.get(c, 0) for c in range(k + 1)},
        })
    stats.update({
        "truncated_rate": round(1 - sum(complete) / max(n_r, 1), 4),
        "no_answer_rate": round(sum(no_answer) / max(n_r, 1), 4),
        "leak_rate": round(sum(leaky) / max(n_r, 1), 4),
    })
    if has_labels:
        n_corr = max(sum(correctness), 1)
        stats["leak_rate_among_correct"] = round(sum(l for l, c in zip(leaky, correctness) if c) / n_corr, 4)
        stats["clean_rate"] = round(sum(clean) / max(n_r, 1), 4)
        stats["clean_questions"] = sum(
            any(clean[i] for i in range(start, start + len(rec["generated_responses"])))
            for rec, start in zip(records, _offsets(records)))

    # lengths
    thinks, finals = zip(*(split_think(r) for r in responses)) if responses else ((), ())
    stats["chars"] = summary([len(r) for r in responses])
    stats["think_chars"] = summary([len(t) for t in thinks])
    stats["final_chars"] = summary([len(f) for f in finals])
    if has_labels:
        stats["chars_correct"] = summary([len(r) for r, c in zip(responses, correctness) if c])
        stats["chars_incorrect"] = summary([len(r) for r, c in zip(responses, correctness) if not c])
    if tokenizer is not None:
        tok = lambda texts: [len(ids) for ids in tokenizer(list(texts), add_special_tokens=False)["input_ids"]]
        lengths = tok(responses)
        stats["tokens"] = summary(lengths)
        stats["think_tokens"] = summary(tok(thinks))
        stats["final_tokens"] = summary(tok(finals))
        if has_labels:
            stats["tokens_correct"] = summary([n for n, c in zip(lengths, correctness) if c])
            stats["tokens_incorrect"] = summary([n for n, c in zip(lengths, correctness) if not c])
    return stats


def _offsets(records):
    start = 0
    for rec in records:
        yield start
        start += len(rec["generated_responses"])


def load_tokenizer(name, cache):
    if name not in cache:
        from transformers import AutoTokenizer
        cache[name] = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    return cache[name]


# (label, how to read it from stats) for the printed table and csv
COLUMNS = [
    ("model", lambda s: s["model"]), ("data", lambda s: s["data"]), ("prompt", lambda s: s["prompt"]),
    ("cfg", lambda s: s["cfg"]), ("#q", lambda s: s["questions"]), ("#resp", lambda s: s["responses"]),
    ("acc", lambda s: s.get("accuracy")), ("solved_any", lambda s: s.get("solved_any")),
    ("solved_all", lambda s: s.get("solved_all")), ("solved_none", lambda s: s.get("solved_none")),
    ("trunc", lambda s: s["truncated_rate"]), ("no_ans", lambda s: s["no_answer_rate"]),
    ("leak", lambda s: s["leak_rate"]), ("clean", lambda s: s.get("clean_rate")),
    ("clean_q", lambda s: s.get("clean_questions")),
    ("chars_med", lambda s: s["chars"]["median"]), ("think_chars_med", lambda s: s["think_chars"]["median"]),
    ("final_chars_med", lambda s: s["final_chars"]["median"]),
    ("tokens_med", lambda s: s.get("tokens", {}).get("median")),
    ("tokens_correct_med", lambda s: s.get("tokens_correct", {}).get("median")),
    ("tokens_incorrect_med", lambda s: s.get("tokens_incorrect", {}).get("median")),
]


def _pct(v):
    return "-" if v is None else f"{100 * v:.1f}%"


def _num(v):
    return "-" if v is None else f"{v:,.0f}"


def _table(title, header, rows):
    """Print one aligned table: first column left-aligned, the rest right-aligned."""
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    fmt = lambda row: "  ".join(str(v).ljust(w) if i == 0 else str(v).rjust(w)
                                for i, (v, w) in enumerate(zip(row, widths)))
    line = fmt(header)
    print(f"\n{title}\n{line}\n{'-' * len(line)}")
    for row in rows:
        print(fmt(row))


def print_report(all_stats):
    # fields shared by all files go in the header instead of every row
    shared = {k: all_stats[0][k] for k in ("data", "prompt", "cfg") if len({s[k] for s in all_stats}) == 1}
    label = lambda s: " | ".join([s["model"]] + [s[k] for k in ("data", "prompt", "cfg") if k not in shared])
    bar = "=" * 72
    print(bar)
    print("RESPONSE STATISTICS  " + "  ".join(f"{k}={v}" for k, v in shared.items()))
    print(bar)

    if any("accuracy" in s for s in all_stats):
        _table("Correctness (share of questions with >=1 / all / no correct responses)",
               ["model", "questions", "responses", "accuracy", "solved >=1", "solved all", "solved none"],
               [[label(s), s["questions"], s["responses"], _pct(s.get("accuracy")), _pct(s.get("solved_any")),
                 _pct(s.get("solved_all")), _pct(s.get("solved_none"))] for s in all_stats])

    _table("Response quality (clean = correct + complete + no hint leak, as convert_to_sft --select clean)",
           ["model", "truncated", "no answer", "hint leak", "leak|correct", "clean", "clean questions"],
           [[label(s), _pct(s["truncated_rate"]), _pct(s["no_answer_rate"]), _pct(s["leak_rate"]),
             _pct(s.get("leak_rate_among_correct")), _pct(s.get("clean_rate")),
             "-" if s.get("clean_questions") is None else f"{s['clean_questions']} / {s['questions']}"]
            for s in all_stats])

    unit = "tokens" if any("tokens" in s for s in all_stats) else "chars"
    _table(f"Median length ({unit})",
           ["model", "response", "think", "final", "correct", "incorrect"],
           [[label(s), _num(s[unit]["median"]), _num(s[f"think_{unit}"]["median"]),
             _num(s[f"final_{unit}"]["median"]), _num(s.get(f"{unit}_correct", {}).get("median")),
             _num(s.get(f"{unit}_incorrect", {}).get("median"))] for s in all_stats])

    hists = [s for s in all_stats if "correct_per_question_hist" in s]
    if hists:
        ks = sorted({k for s in hists for k in s["correct_per_question_hist"]}, key=int)
        _table("Questions by number of correct responses",
               ["model"] + [f"{k} correct" for k in ks],
               [[label(s)] + [s["correct_per_question_hist"].get(k, 0) for k in ks] for s in hists])


def main():
    args = parse_args()
    tokenizers = {}
    all_stats = []
    for path in args.input_files:
        tokenizer = None
        if args.count_tokens:
            name = args.tokenizer or describe_path(path)["model"]
            tokenizer = load_tokenizer(name, tokenizers)
        all_stats.append(file_stats(path, tokenizer))
    print_report(all_stats)
    cols = [(name, get) for name, get in COLUMNS if any(get(s) is not None for s in all_stats)]

    os.makedirs(args.output_dir, exist_ok=True)
    if args.output_name is None:
        d = all_stats[0]
        args.output_name = (f"{d['data']}_{d['prompt']}_{d['cfg']}" if len(all_stats) > 1
                            else f"{d['data']}_{d['model'].replace('/', '_')}_{d['prompt']}_{d['cfg']}")
    json_path = os.path.join(args.output_dir, f"{args.output_name}.json")
    csv_path = os.path.join(args.output_dir, f"{args.output_name}.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([name for name, _ in cols])
        writer.writerows([[get(s) for _, get in cols] for s in all_stats])
    print(f"\nsaved: {json_path}\n       {csv_path}")


if __name__ == "__main__":
    main()
