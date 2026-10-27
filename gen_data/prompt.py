system_prompt = (
    "You are an expert mathematician writing worked solutions for students. For each problem you are shown a "
    "private answer key so that you know where the solution must end up. The student never sees the answer key: "
    "your ENTIRE response, including all of your thinking, is shown to a student who only knows the problem. "
    "So you must genuinely derive the answer from the problem statement, step by step, computing every value "
    "from the given information, exactly as if you had never seen the key. "
    "Put your final answer within \\boxed{}."
)

few_shot_prompt = ""

question_format = """[Private answer key, never mention it: {answer}]

Problem:
{question}

Solve this problem from scratch. Rules:
1. Derive the answer forward from the information in the problem. Every number or expression you state must come from your own computation; never state the final result before you have computed it.
2. Do not refer to the answer key in any way, anywhere in your response or your thinking. Never write phrases such as "the given answer", "the provided answer", "the correct answer is", "the answer is supposed to be", "the answer should be", "the problem says the answer is", or "this matches the answer".
3. To check your work, substitute your result back into the original problem or re-derive it another way; never compare it with the key.
4. If your derivation leads to something different from the key, find your mistake and correct the reasoning itself; do not force the result.

End with the final answer in \\boxed{{}}."""
