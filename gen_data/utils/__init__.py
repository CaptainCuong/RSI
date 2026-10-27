"""Answer extraction and grading, copied from SLM_REASONING/eval/utils (parser, grader, math_normalization).

A regular package with relative imports only, so it cannot be shadowed by another `utils` on sys.path.
"""
from .parser import extract_answer, strip_string
from .grader import check_is_correct, math_equal

__all__ = ["extract_answer", "strip_string", "check_is_correct", "math_equal"]
