"""Evaluate a model on math benchmarks with vLLM (port of SLM_REASONING/eval/eval.py).

Loads the model once and evaluates every benchmark in --data_names (eval/data/<name>/test.jsonl), or a
single file given by --data_file. Answers are extracted and graded with gen_data/utils in parallel
worker processes. Per-benchmark generations are saved as jsonl (resumable: finished questions are
skipped on rerun) and all metrics are merged into one metrics json per model and sampling config.

Metrics: avg@n (mean accuracy over all n samples), pass@k (unbiased estimator), maj@n (majority vote).

Example:
    CUDA_VISIBLE_DEVICES=0 python eval/eval.py --model_name_or_path Qwen/Qwen3-0.6B \
        --data_names math,aime,amc --n_sampling 8 --k 1,8 --temperature 0.6 --top_p 0.95
"""
import argparse
import json
import multiprocessing
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from math import comb

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "..", "gen_data"))
from generate import load_prompt, load_examples, get_question, load_done, set_seed, _init_grader, _grade  # noqa: E402

# benchmarks whose `answer` field is not the bare final answer
ANSWER_KEYS = {"gaokao": "choice_answer"}  # gaokao `answer` is e.g. "C. {x | -3<x<4}"


def parse_list(arg):
    return [x for x in arg.split(",") if x]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_names", type=parse_list, default=["math"],
                        help="comma-separated benchmarks under --data_dir, e.g. math,aime,amc")
    parser.add_argument("--data_dir", type=str, default=os.path.join(THIS_DIR, "data"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--data_file", type=str, default=None,
                        help="evaluate this .json/.jsonl file instead of --data_names")
    parser.add_argument("--question_key", type=str, default=None, help="default: first of question/problem/Question/input")
    parser.add_argument("--answer_key", type=str, default=None, help="default: 'answer' (see ANSWER_KEYS)")
    parser.add_argument("--prompt_file", type=str, default=os.path.join(THIS_DIR, "prompt.py"))
    parser.add_argument("--output_dir", type=str, default="./outputs/eval")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1, help="data[start:end], if -1, data[start:]")
    parser.add_argument("--n_sampling", type=int, default=1)
    parser.add_argument("--k", type=lambda s: [int(x) for x in parse_list(s)], default=[1],
                        help="comma-separated k values for pass@k, each <= n_sampling")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--stop", type=parse_list, default=None)
    parser.add_argument("--use_few_shot", action="store_true")
    parser.add_argument("--no_chat_template", action="store_true", help="feed raw prompt instead of chat messages")
    parser.add_argument("--disable_thinking", action="store_true",
                        help="pass enable_thinking=False to the chat template (Qwen3-style models)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=min(32, os.cpu_count() or 1))
    args = parser.parse_args()

    args.top_p = 1 if args.temperature == 0 else args.top_p  # top_p must be 1 when using greedy
    bad_k = [k for k in args.k if k > args.n_sampling]
    if bad_k:
        parser.error(f"--k {bad_k} larger than --n_sampling {args.n_sampling}")
    return args


def pass_at_k(n, c, k):
    """Unbiased pass@k from n samples with c correct (Chen et al., 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def majority_correct(answers, correctness):
    votes = Counter(a.strip() for a in answers if a and a.strip())
    if not votes:
        return False
    top = votes.most_common(1)[0][0]
    return next(c for a, c in zip(answers, correctness) if a and a.strip() == top)


def compute_metrics(records, ks):
    n = len(records)
    metrics = {"num_questions": n}
    if n == 0:
        return metrics
    metrics[f"avg@{len(records[0]['answers_correctness'])}"] = (
        sum(sum(r["answers_correctness"]) / len(r["answers_correctness"]) for r in records) / n)
    for k in ks:
        metrics[f"pass@{k}"] = sum(
            pass_at_k(len(r["answers_correctness"]), sum(r["answers_correctness"]), k) for r in records) / n
    metrics[f"maj@{len(records[0]['answers_correctness'])}"] = (
        sum(majority_correct(r["generated_answers"], r["answers_correctness"]) for r in records) / n)
    metrics["avg_response_chars"] = sum(
        sum(map(len, r["generated_responses"])) / len(r["generated_responses"]) for r in records) / n
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}


def main(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if args.data_file is not None:
        benchmarks = [(os.path.splitext(os.path.basename(args.data_file))[0], args.data_file)]
    else:
        benchmarks = [(name, os.path.join(args.data_dir, name, f"{args.split}.jsonl")) for name in args.data_names]
    for _, path in benchmarks:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    model_name = "/".join(args.model_name_or_path.rstrip("/").split("/")[-3:])
    cfg = f"{args.split}_t{args.temperature}_n{args.n_sampling}" + ("_nothink" if args.disable_thinking else "")
    out_root = os.path.join(args.output_dir, model_name)

    # collect the questions still to generate for every benchmark
    jobs = []
    for name, path in benchmarks:
        examples = load_examples(path)
        end_idx = len(examples) if args.end_idx == -1 else args.end_idx
        answer_key = args.answer_key or ANSWER_KEYS.get(name, "answer")
        out_file = os.path.join(out_root, name, f"{cfg}_s{args.start_idx}_e{end_idx}.jsonl")
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        indexed = list(enumerate(examples))[args.start_idx:end_idx]
        done = {d["idx"] for d in load_done(out_file)}
        todo = [(i, d) for i, d in indexed if i not in done]
        print(f"[{name}] {len(indexed) - len(todo)}/{len(indexed)} already evaluated, {len(todo)} remaining")
        jobs.append(dict(name=name, out_file=out_file, answer_key=answer_key, todo=todo))

    if any(job["todo"] for job in jobs):
        system_prompt, few_shot_prompt, question_format = load_prompt(args.prompt_file)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        template_kwargs = {"enable_thinking": False} if args.disable_thinking else {}

        def build_prompt(example):
            cur_prompt = question_format.format(question=get_question(example, args.question_key))
            if args.use_few_shot:
                cur_prompt = few_shot_prompt + cur_prompt
            if args.no_chat_template:
                return cur_prompt
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": cur_prompt}]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 **template_kwargs)

        # spawn: the parent holds CUDA state; created before the model so import errors surface immediately
        grader_pool = ProcessPoolExecutor(max_workers=args.num_workers,
                                          mp_context=multiprocessing.get_context("spawn"),
                                          initializer=_init_grader)
        assert grader_pool.submit(_grade, ("\\boxed{\\frac{1}{2}}", "0.5")).result() == ("\\frac{1}{2}", True)

        available_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
        llm_kwargs = dict(model=args.model_name_or_path, tensor_parallel_size=len(available_gpus),
                          trust_remote_code=True, gpu_memory_utilization=args.gpu_memory_utilization,
                          dtype=args.dtype, seed=args.seed)
        if args.max_model_len is not None:
            llm_kwargs["max_model_len"] = args.max_model_len
        llm = LLM(**llm_kwargs)
        sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
                                         n=args.n_sampling, stop=args.stop, seed=args.seed)

        for job in jobs:
            if not job["todo"]:
                continue
            prompts = [build_prompt(d) for _, d in job["todo"]]
            print(f"[{job['name']}] generating {len(prompts)} x {args.n_sampling}\n{prompts[0]}")
            t0 = time.time()
            completions = llm.generate(prompts, sampling_params)
            records = []
            for (i, d), completion in zip(job["todo"], completions):
                records.append({
                    "idx": i,
                    "question": get_question(d, args.question_key),
                    "gold_answer": str(d[job["answer_key"]]),
                    "generated_responses": [o.text for o in completion.outputs],
                    "finish_reasons": [o.finish_reason for o in completion.outputs],
                    **({"id": d["id"]} if "id" in d else {}),
                })
            futures = [grader_pool.submit(_grade, (resp, r["gold_answer"]))
                       for r in records for resp in r["generated_responses"]]
            results = iter([f.result() for f in futures])
            with open(job["out_file"], "a", encoding="utf-8") as f:
                for r in records:
                    graded = [next(results) for _ in r["generated_responses"]]
                    r["generated_answers"] = [a for a, _ in graded]
                    r["answers_correctness"] = [c for _, c in graded]
                    r["is_correct"] = any(r["answers_correctness"])
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[{job['name']}] done in {time.time() - t0:.0f}s")
        grader_pool.shutdown()

    # metrics over everything in the output files (including runs resumed from earlier)
    metrics_file = os.path.join(out_root, f"metrics_{cfg}.json")
    all_metrics = {}
    if os.path.exists(metrics_file):
        with open(metrics_file, encoding="utf-8") as f:
            all_metrics = json.load(f)
    for job in jobs:
        records = load_done(job["out_file"])
        all_metrics[job["name"]] = compute_metrics(records, args.k)
        truncated = sum(fr == "length" for r in records for fr in r.get("finish_reasons", []))
        all_metrics[job["name"]]["truncated_responses"] = truncated
    all_metrics = dict(sorted(all_metrics.items()))
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    cols = [c for c in next(iter(all_metrics.values())) if c.startswith(("avg@", "pass@", "maj@"))]
    print(f"\nmodel: {args.model_name_or_path}  ({cfg})")
    print(f"{'benchmark':20s}{'#q':>6s}" + "".join(f"{c:>10s}" for c in cols) + f"{'trunc':>8s}")
    for name, m in all_metrics.items():
        print(f"{name:20s}{m['num_questions']:>6d}" + "".join(f"{m.get(c, float('nan')):>10.4f}" for c in cols)
              + f"{m.get('truncated_responses', 0):>8d}")
    print(f"saved metrics to {metrics_file}")


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)
