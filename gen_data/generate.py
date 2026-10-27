"""Generate answer-guided solutions for a dataset with vLLM.

Each sample is given the problem *and* its gold answer (see gen_data/prompt.py),
and the model is asked to write a solution that leads to that answer. The
generated answers are then checked against the gold answer so that only
consistent solutions can be kept downstream.

Example:
    CUDA_VISIBLE_DEVICES=0,1 python gen_data/generate.py \
        --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
        --input_file data/math12k_train.json \
        --output_file outputs/math12k_train_gen.jsonl \
        --n_sampling 4 --temperature 0.7 --top_p 0.95
"""
import argparse
import importlib.util
import json
import multiprocessing
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_list(arg):
    return arg.split(',')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True, help="model dir or HF id")
    parser.add_argument('--input_file', type=str, default="data/math12k_train.json", help=".json list or .jsonl")
    parser.add_argument('--output_file', type=str, default=None, help="output .jsonl (default: derived from input/model)")
    parser.add_argument('--output_dir', type=str, default="./outputs")
    parser.add_argument('--prompt_file', type=str, default=os.path.join(THIS_DIR, "prompt.py"))
    parser.add_argument('--prompt_type', type=str, default=None,
                        help="name used in the output path (default: prompt file name, e.g. prompt_simple)")
    parser.add_argument('--question_key', type=str, default=None, help="default: first of question/problem/Question/input")
    parser.add_argument('--answer_key', type=str, default="answer")
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1, help="data[start:end], if -1, data[start:]")
    parser.add_argument('--n_sampling', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--max_tokens', type=int, default=8192)
    parser.add_argument('--stop', type=parse_list, default=None)
    parser.add_argument('--use_few_shot', action="store_true")
    parser.add_argument('--no_chat_template', action="store_true", help="feed raw prompt instead of chat messages")
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    parser.add_argument('--max_model_len', type=int, default=None)
    parser.add_argument('--dtype', type=str, default="auto")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no_check', action="store_true", help="skip checking generated answers against gold")
    parser.add_argument('--chunk_size', type=int, default=500,
                        help="generate and append results every chunk_size samples (enables resuming)")
    parser.add_argument('--num_workers', type=int, default=min(32, os.cpu_count() or 1),
                        help="processes for answer extraction/grading (overlaps with generation of the next chunk)")
    args = parser.parse_args()

    args.top_p = 1 if args.temperature == 0 else args.top_p  # top_p must be 1 when using greedy
    if args.prompt_type is None:
        args.prompt_type = os.path.splitext(os.path.basename(args.prompt_file))[0]
    return args


def set_seed(seed):
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_prompt(prompt_file):
    spec = importlib.util.spec_from_file_location("gen_prompt", prompt_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ["system_prompt", "few_shot_prompt", "question_format"]:
        if not hasattr(module, name):
            raise AttributeError(f"'{name}' not found in {prompt_file}")
    return module.system_prompt, module.few_shot_prompt, module.question_format


def load_examples(path):
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_question(example, question_key=None):
    if question_key is not None:
        return example[question_key].strip()
    for key in ["question", "problem", "Question", "input"]:
        if key in example:
            return example[key].strip()
    raise KeyError(f"No question field found in example with keys {list(example.keys())}")


def _init_grader():
    global extract_answer, check_is_correct
    # one BLAS/OpenMP thread per worker; many workers each spinning up a full thread pool oversubscribes the CPU
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[var] = "1"
    # gen_data/utils (answer extraction + grading); put it first so no other `utils` can shadow it
    sys.path.insert(0, THIS_DIR)
    from utils import extract_answer, check_is_correct


def _grade(args):
    response, gold = args
    answer = ""
    try:
        answer = extract_answer(response)
        return answer, bool(check_is_correct(answer, gold))
    except Exception:
        return answer, False


def load_done(output_file):
    """Read existing results; drop a truncated last line left by a crash and rewrite the file cleanly."""
    if not os.path.exists(output_file):
        return []
    records = []
    with open(output_file, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"dropping malformed line in {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        for d in records:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return records


def report_completion(args, start_time, n_generated):
    """Print a summary banner; called last so it is not buried under vLLM's shutdown logs."""
    records = load_done(args.output_file)
    lines = [
        "GENERATION COMPLETE",
        f"model:   {args.model_name_or_path}",
        f"data:    {args.input_file} [{args.start_idx}:{args.end_idx}]",
        f"prompt:  {args.prompt_file}",
        f"output:  {args.output_file}",
        f"samples: {len(records)} saved ({n_generated} generated in this run)",
    ]
    if not args.no_check and records and "answers_correctness" in records[0]:
        correct_cnt = sum(sum(d["answers_correctness"]) for d in records)
        total_cnt = sum(len(d["answers_correctness"]) for d in records)
        n_any = sum(d["is_correct"] for d in records)
        lines.append(f"response-level consistency with gold: {correct_cnt}/{total_cnt} = {correct_cnt / max(total_cnt, 1):.4f}")
        lines.append(f"samples with >=1 consistent response: {n_any}/{len(records)} = {n_any / len(records):.4f}")
    lines.append(f"elapsed: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
    bar = "=" * max(len(line) for line in lines)
    print("\n".join(["", bar, *lines, bar]), flush=True)


def infer(args):
    start_time = time.time()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    examples = load_examples(args.input_file)
    if args.end_idx == -1:
        args.end_idx = len(examples)
    # keep the original index so finished samples can be identified when resuming
    indexed = list(enumerate(examples))[args.start_idx:args.end_idx]

    if args.output_file is None:
        model_name = "/".join(args.model_name_or_path.rstrip("/").split("/")[-3:])
        data_name = os.path.splitext(os.path.basename(args.input_file))[0]
        args.output_file = os.path.join(
            args.output_dir, model_name, data_name, args.prompt_type,
            f"t{args.temperature}_k{args.n_sampling}_s{args.start_idx}_e{args.end_idx}.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    done_idx = {d["idx"] for d in load_done(args.output_file)}
    todo = [(i, d) for i, d in indexed if i not in done_idx]
    print(f"{len(indexed) - len(todo)}/{len(indexed)} samples already generated in {args.output_file}, "
          f"{len(todo)} remaining")
    if not todo:
        report_completion(args, start_time, 0)
        return

    # Same batching scheme as SLM_REASONING/eval/eval.py: n per request <= 64, repeated over epochs
    factor = 1
    for i in range(2, 65):
        if args.n_sampling % i == 0:
            factor = i
    generation_epoch = args.n_sampling // factor
    print(f"use n = {factor}, generation epoch is: {generation_epoch}")

    system_prompt, few_shot_prompt, question_format = load_prompt(args.prompt_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    def build_prompt(example):
        question = get_question(example, args.question_key)
        answer = str(example[args.answer_key])
        cur_prompt = question_format.format(question=question, answer=answer)
        if args.use_few_shot:
            cur_prompt = few_shot_prompt + cur_prompt
        if not args.no_chat_template:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": cur_prompt},
            ]
            cur_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return cur_prompt

    print(build_prompt(todo[0][1]))

    grader_pool = None
    if not args.no_check:
        # spawn (not fork): the parent process holds CUDA/vLLM state. The executor's workers are
        # non-daemonic, so the grader can still start its own timeout subprocesses inside them.
        grader_pool = ProcessPoolExecutor(
            max_workers=args.num_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_grader,
        )
        # fail fast (before loading the model) if the grader cannot be imported in the workers
        assert grader_pool.submit(_grade, ("\\boxed{\\frac{1}{2}}", "0.5")).result() == ("\\frac{1}{2}", True)

    available_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')
    print(f"available_gpus: {available_gpus}")
    llm_kwargs = dict(
        model=args.model_name_or_path,
        tensor_parallel_size=len(available_gpus),
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        seed=args.seed,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)

    def submit_grading(outputs):
        if grader_pool is None:
            return None
        tasks = [(r, out["gold_answer"]) for out in outputs for r in out["generated_responses"]]
        return [grader_pool.submit(_grade, t) for t in tasks]

    def finish_chunk(outputs, futures):
        if futures is not None:
            results = iter([f.result() for f in futures])
            for out in outputs:
                graded = [next(results) for _ in out["generated_responses"]]
                out["generated_answers"] = [a for a, _ in graded]
                out["answers_correctness"] = [c for _, c in graded]
                out["is_correct"] = any(out["answers_correctness"])
        with open(args.output_file, "a", encoding="utf-8") as f:
            for out in outputs:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    pending = None  # (outputs, futures) of the previous chunk, graded while the next chunk generates
    for start in range(0, len(todo), args.chunk_size):
        chunk = todo[start:start + args.chunk_size]
        print(f"chunk {start // args.chunk_size + 1}/{(len(todo) - 1) // args.chunk_size + 1}: {len(chunk)} samples")
        prompt_batch = [build_prompt(d) for _, d in chunk]
        outputs = [{
            "idx": i,
            "question": get_question(d, args.question_key),
            "gold_answer": str(d[args.answer_key]),
            "generated_responses": [],
            **({"id": d["id"]} if "id" in d else {}),
            **({"source": d["source"]} if "source" in d else {}),
        } for i, d in chunk]

        for epoch in range(generation_epoch):
            sampling_params = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                n=factor,
                stop=args.stop,
                seed=args.seed + epoch,
            )
            completions = llm.generate(prompt_batch, sampling_params)
            for out, completion in zip(outputs, completions):
                out["generated_responses"] += [o.text for o in completion.outputs]

        if pending is not None:
            finish_chunk(*pending)
        pending = (outputs, submit_grading(outputs))
    if pending is not None:
        finish_chunk(*pending)
    if grader_pool is not None:
        grader_pool.shutdown()
    # shut vLLM down now (instead of at interpreter exit) so its teardown logs come before the summary
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception as e:
        print(f"vLLM shutdown failed (ignored): {e}")
    report_completion(args, start_time, len(todo))


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    infer(args)
