#!/usr/bin/env python3
"""
Local Coder / Bug-Fixer Benchmark — LM Studio backend
=====================================================
Runs the same HumanEval / MBPP / BugFix / Summary / InfoPres benchmark
suite against every LLM you have downloaded inside LM Studio, one at a
time. For each model the script:

    1. Asks LM Studio (via the `lms` CLI) to load the model.
    2. Hits LM Studio's OpenAI-compatible REST API to run all tasks.
    3. Asks `lms` to unload the model before moving on.

Requirements:
    - LM Studio installed locally with the `lms` CLI on your $PATH
      (run `lms bootstrap` once if it isn't).
    - The models you want to test already downloaded in LM Studio.
    - LM Studio's local server reachable at http://127.0.0.1:1234
      (the default). The script will start the server for you if it
      isn't already running.

Usage:
    python gemma4_senior_bench.py              # full run, all LM Studio LLMs
    python gemma4_senior_bench.py --dry-run    # list discovered models, no benchmark
    python gemma4_senior_bench.py --report     # reprint report from saved results
    python gemma4_senior_bench.py --models a,b # only test models whose key contains a or b
    python gemma4_senior_bench.py --host 127.0.0.1 --port 1234

Resumes automatically if interrupted (skips already-completed models).
"""

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from textwrap import indent
from typing import Dict, List, Optional

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────

LM_STUDIO_HOST = "127.0.0.1"
LM_STUDIO_PORT = 1234
LM_STUDIO_URL  = f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}"
SERVER_TIMEOUT = 120     # seconds to wait for LM Studio server to come up
LOAD_TIMEOUT   = 600     # seconds to wait for a model to finish loading

RESULTS_DIR  = Path(__file__).parent / "results"

# Composite score weights. This benchmark now prioritizes local coder/bug-fixer
# quality first, with summarization kept as a secondary handoff capability.
WEIGHTS = {"humaneval": 0.24, "mbpp": 0.24, "bugfix": 0.24, "summary": 0.16, "infopres": 0.12}
RUN_PREFIX = "coderbench"

# ─── MODELS ────────────────────────────────────────────────────────────────────
# The model list is discovered at runtime from LM Studio (`lms ls --json`).
# See discover_lm_studio_models() further down. Nothing is hard-coded here.

# ─── HUMANEVAL PROBLEMS (20) ───────────────────────────────────────────────────
# Source: OpenAI HumanEval (MIT License), representative subset

HUMANEVAL = [
    {
        "task_id": "HE/1",
        "prompt": 'def has_close_elements(numbers: list, threshold: float) -> bool:\n    """Check if in given list of numbers, are any two numbers closer to each\n    other than given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
        "test": "assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\nassert has_close_elements([1.0, 2.0, 3.0], 1.5) == True\nassert has_close_elements([1.0, 2.0, 3.0], 3.0) == True",
    },
    {
        "task_id": "HE/3",
        "prompt": 'def truncate_number(number: float) -> float:\n    """Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    """\n',
        "test": "assert abs(truncate_number(3.5) - 0.5) < 1e-6\nassert abs(truncate_number(1.25) - 0.25) < 1e-6\nassert abs(truncate_number(123.0) - 0.0) < 1e-6",
    },
    {
        "task_id": "HE/4",
        "prompt": 'def below_zero(operations: list) -> bool:\n    """You\'re given a list of deposit and withdrawal operations on a bank account\n    that starts with zero balance. Your task is to detect if at any point the\n    balance of account falls below zero, and at that point function should\n    return True. Otherwise it should return False.\n    >>> below_zero([1, 2, 3])\n    False\n    >>> below_zero([1, 2, -4, 5])\n    True\n    """\n',
        "test": "assert below_zero([1, 2, 3]) == False\nassert below_zero([1, 2, -4, 5]) == True\nassert below_zero([]) == False\nassert below_zero([-1]) == True",
    },
    {
        "task_id": "HE/6",
        "prompt": 'def parse_nested_parens(paren_string: str) -> list:\n    """Input to this function is a string represented multiple groups for nested\n    parentheses separated by spaces. For each of the group, output the deepest\n    level of nesting of parentheses.\n    E.g. (()()) has maximum two levels of nesting while ((())) has three.\n    >>> parse_nested_parens(\'(()()) ((())) () ((())(()()))\')\n    [2, 3, 1, 3]\n    """\n',
        "test": "assert parse_nested_parens('(()()) ((())) () ((())(()()))') == [2, 3, 1, 3]\nassert parse_nested_parens('() (()) ((())) (((())))') == [1, 2, 3, 4]",
    },
    {
        "task_id": "HE/7",
        "prompt": 'def filter_by_substring(strings: list, substring: str) -> list:\n    """Filter an input list of strings only for ones that contain given substring\n    >>> filter_by_substring([], \'a\')\n    []\n    >>> filter_by_substring([\'abc\', \'bacd\', \'cde\', \'array\'], \'a\')\n    [\'abc\', \'bacd\', \'array\']\n    """\n',
        "test": "assert filter_by_substring([], 'a') == []\nassert filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a') == ['abc', 'bacd', 'array']\nassert filter_by_substring(['hello', 'world'], 'xyz') == []",
    },
    {
        "task_id": "HE/9",
        "prompt": 'def rolling_max(numbers: list) -> list:\n    """From a given list of integers, generate a list of rolling maximum element\n    found until given moment in the sequence.\n    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])\n    [1, 2, 3, 3, 3, 4, 4]\n    """\n',
        "test": "assert rolling_max([1, 2, 3, 2, 3, 4, 2]) == [1, 2, 3, 3, 3, 4, 4]\nassert rolling_max([]) == []\nassert rolling_max([5, 4, 3, 2, 1]) == [5, 5, 5, 5, 5]",
    },
    {
        "task_id": "HE/11",
        "prompt": 'def string_xor(a: str, b: str) -> str:\n    """Input are two strings a and b consisting only of 1s and 0s.\n    Perform binary XOR on these inputs and return result also as a string.\n    >>> string_xor(\'010\', \'110\')\n    \'100\'\n    """\n',
        "test": "assert string_xor('010', '110') == '100'\nassert string_xor('0', '1') == '1'\nassert string_xor('111', '111') == '000'",
    },
    {
        "task_id": "HE/13",
        "prompt": 'def greatest_common_divisor(a: int, b: int) -> int:\n    """Return a greatest common divisor of two integers a and b\n    >>> greatest_common_divisor(3, 5)\n    1\n    >>> greatest_common_divisor(25, 15)\n    5\n    """\n',
        "test": "assert greatest_common_divisor(3, 5) == 1\nassert greatest_common_divisor(25, 15) == 5\nassert greatest_common_divisor(0, 5) == 5\nassert greatest_common_divisor(12, 8) == 4",
    },
    {
        "task_id": "HE/14",
        "prompt": 'def all_prefixes(string: str) -> list:\n    """Return list of all prefixes from shortest to longest of the input string\n    >>> all_prefixes(\'abc\')\n    [\'a\', \'ab\', \'abc\']\n    """\n',
        "test": "assert all_prefixes('abc') == ['a', 'ab', 'abc']\nassert all_prefixes('') == []\nassert all_prefixes('x') == ['x']",
    },
    {
        "task_id": "HE/17",
        "prompt": 'def parse_music(music_string: str) -> list:\n    """Input to this function is a string representing musical notes in a special\n    ASCII format. Your task is to parse this string and return list of integers\n    corresponding to how many beats does each not last.\n    Here is a legend:\n    \'o\' - whole note, lasts four beats\n    \'o|\' - half note, lasts two beats\n    \'.|\'- quarter note, lasts one beat\n    >>> parse_music(\'o o| .| o| o| .| .| .| .| o o\')\n    [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]\n    """\n',
        "test": "assert parse_music('o o| .| o| o| .| .| .| .| o o') == [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]\nassert parse_music('') == []\nassert parse_music('o') == [4]",
    },
    {
        "task_id": "HE/18",
        "prompt": 'def how_many_times(string: str, substring: str) -> int:\n    """Find how many times a given substring can be found in the original string.\n    Count overlapping cases.\n    >>> how_many_times(\'\', \'a\')\n    0\n    >>> how_many_times(\'aaa\', \'a\')\n    3\n    >>> how_many_times(\'aaaa\', \'aa\')\n    3\n    """\n',
        "test": "assert how_many_times('', 'a') == 0\nassert how_many_times('aaa', 'a') == 3\nassert how_many_times('aaaa', 'aa') == 3",
    },
    {
        "task_id": "HE/21",
        "prompt": 'def rescale_to_unit(numbers: list) -> list:\n    """Given list of numbers (of at least two elements), apply a linear transform\n    to that list, such that the smallest number will become 0 and the largest\n    will become 1\n    >>> rescale_to_unit([1.0, 2.0, 3.0, 4.0, 5.0])\n    [0.0, 0.25, 0.5, 0.75, 1.0]\n    """\n',
        "test": "result = rescale_to_unit([1.0, 2.0, 3.0, 4.0, 5.0])\nassert all(abs(a-b)<1e-6 for a,b in zip(result, [0.0, 0.25, 0.5, 0.75, 1.0]))\nresult2 = rescale_to_unit([0, 10])\nassert abs(result2[0]) < 1e-6 and abs(result2[1]-1.0) < 1e-6",
    },
    {
        "task_id": "HE/26",
        "prompt": 'def remove_duplicates(numbers: list) -> list:\n    """From a list of integers, remove all elements that occur more than once.\n    Keep order of elements left the same as in the input.\n    >>> remove_duplicates([1, 2, 3, 2, 4])\n    [1, 3, 4]\n    """\n',
        "test": "assert remove_duplicates([1, 2, 3, 2, 4]) == [1, 3, 4]\nassert remove_duplicates([]) == []\nassert remove_duplicates([1, 1, 1]) == []",
    },
    {
        "task_id": "HE/28",
        "prompt": 'def concatenate(strings: list) -> str:\n    """Concatenate list of strings into a single string\n    >>> concatenate([])\n    \'\'\n    >>> concatenate([\'a\', \'b\', \'c\'])\n    \'abc\'\n    """\n',
        "test": "assert concatenate([]) == ''\nassert concatenate(['a', 'b', 'c']) == 'abc'\nassert concatenate(['hello', ' ', 'world']) == 'hello world'",
    },
    {
        "task_id": "HE/32",
        "prompt": 'def find_zero(xs: list) -> float:\n    """xs are coefficients of a polynomial.\n    find_zero find x such that poly(x) = 0.\n    find_zero returns only one zero point, even if there are many.\n    Moreover, find_zero only takes list xs having even number of coefficients\n    and the largest non zero coefficient as it guarantees a solution.\n    >>> round(find_zero([1, 2]), 2)\n    -0.5\n    >>> round(find_zero([-6, 11, -6, 1]), 2)\n    1.0\n    """\n    def poly(xs, x):\n        return sum([coeff * (x ** i) for i, coeff in enumerate(xs)])\n',
        "test": "def poly(xs, x): return sum([c*(x**i) for i,c in enumerate(xs)])\nassert abs(poly([1,2], find_zero([1,2]))) < 1e-4\nassert abs(poly([-6,11,-6,1], find_zero([-6,11,-6,1]))) < 1e-4",
    },
    {
        "task_id": "HE/33",
        "prompt": 'def sort_third(l: list) -> list:\n    """This function takes a list l and returns a list l\' such that\n    l\' is identical to l in the indicies that are not divisible by three, while\n    its values at the indicies that are divisible by three are equal\n    to the values of the corresponding indicies of l, but sorted.\n    >>> sort_third([1, 2, 3])\n    [1, 2, 3]\n    >>> sort_third([5, 6, 3, 4, 8, 9, 2])\n    [2, 6, 3, 4, 8, 9, 5]\n    """\n',
        "test": "assert sort_third([1, 2, 3]) == [1, 2, 3]\nassert sort_third([5, 6, 3, 4, 8, 9, 2]) == [2, 6, 3, 4, 8, 9, 5]",
    },
    {
        "task_id": "HE/38",
        "prompt": 'def encode_cyclic(s: str) -> str:\n    """\n    returns encoded string by cycling groups of three characters.\n    """\n    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]\n    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]\n    return "".join(groups)\n\n\ndef decode_cyclic(s: str) -> str:\n    """\n    takes as input string encoded with encode_cyclic function. Returns decoded string.\n    """\n',
        "test": "assert decode_cyclic(encode_cyclic('abc')) == 'abc'\nassert decode_cyclic(encode_cyclic('hello world')) == 'hello world'\nassert decode_cyclic(encode_cyclic('')) == ''",
    },
    {
        "task_id": "HE/42",
        "prompt": 'def incr_list(l: list) -> list:\n    """Return list with elements incremented by 1.\n    >>> incr_list([1, 2, 3])\n    [2, 3, 4]\n    >>> incr_list([5, 3, 5, 2, 3, 3, 9, 0, 123])\n    [6, 4, 6, 3, 4, 4, 10, 1, 124]\n    """\n',
        "test": "assert incr_list([1, 2, 3]) == [2, 3, 4]\nassert incr_list([5, 3, 5, 2, 3, 3, 9, 0, 123]) == [6, 4, 6, 3, 4, 4, 10, 1, 124]\nassert incr_list([]) == []",
    },
    {
        "task_id": "HE/56",
        "prompt": 'def correct_bracketing(brackets: str) -> bool:\n    """brackets is a string of \'<\' and \'>\'. return True if every opening\n    bracket has a corresponding closing bracket.\n    >>> correct_bracketing(\'<\')\n    False\n    >>> correct_bracketing(\'<>\')\n    True\n    >>> correct_bracketing(\'<<><>>\')\n    True\n    >>> correct_bracketing(\'><<>\')\n    False\n    """\n',
        "test": "assert correct_bracketing('<') == False\nassert correct_bracketing('<>') == True\nassert correct_bracketing('<<><>>') == True\nassert correct_bracketing('><<>') == False\nassert correct_bracketing('') == True",
    },
    {
        "task_id": "HE/62",
        "prompt": 'def derivative(xs: list) -> list:\n    """xs represent coefficients of a polynomial.\n    xs[0] + xs[1] * x + xs[2] * x^2 + ....\n    Return derivative of this polynomial in the same form.\n    >>> xs = [3, 1, 2, 4, 5]\n    >>> x = 2\n    >>> derivative(xs)\n    [1, 4, 12, 20]\n    """\n',
        "test": "assert derivative([3, 1, 2, 4, 5]) == [1, 4, 12, 20]\nassert derivative([1]) == []\nassert derivative([0, 1]) == [1]",
    },
]

# ─── MBPP PROBLEMS (20) ────────────────────────────────────────────────────────
# Source: Google MBPP sanitized split (Apache 2.0 License), representative subset

MBPP = [
    {
        "task_id": "MBPP/2",
        "prompt": "Write a Python function to find the similar elements from the given two lists.",
        "test": "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == {4, 5}\nassert similar_elements((1, 2, 3, 4),(5, 4, 3)) == {3, 4}",
        "fn_name": "similar_elements",
    },
    {
        "task_id": "MBPP/3",
        "prompt": "Write a Python function to check whether the given number is odd or not, return True if odd.",
        "test": "assert is_odd(7) == True\nassert is_odd(8) == False\nassert is_odd(1) == True",
        "fn_name": "is_odd",
    },
    {
        "task_id": "MBPP/7",
        "prompt": "Write a Python function to check whether a string is a pangram or not (a pangram contains every letter of the alphabet at least once).",
        "test": "assert is_pangram('The quick brown fox jumps over the lazy dog') == True\nassert is_pangram('hello world') == False",
        "fn_name": "is_pangram",
    },
    {
        "task_id": "MBPP/11",
        "prompt": "Write a Python function to remove first and last occurrence of a given character from the string.",
        "test": "assert remove_Occ('hello','l') == 'helo'\nassert remove_Occ('abcda','a') == 'bcd'",
        "fn_name": "remove_Occ",
    },
    {
        "task_id": "MBPP/12",
        "prompt": "Write a Python function to sort a list of tuples in increasing order by the last element in each tuple.",
        "test": "assert sort_matrix([(1,3),(4,1),(2,2)]) == [(4,1),(2,2),(1,3)]\nassert sort_matrix([(1,2,3),(4,5,6)]) == [(1,2,3),(4,5,6)]",
        "fn_name": "sort_matrix",
    },
    {
        "task_id": "MBPP/14",
        "prompt": "Write a Python function to find the volume of a cylinder given radius and height.",
        "test": "import math\nassert abs(volume_cylinder(10,5) - 1570.796) < 0.01\nassert abs(volume_cylinder(4,5) - math.pi*4*4*5) < 0.01",
        "fn_name": "volume_cylinder",
    },
    {
        "task_id": "MBPP/16",
        "prompt": "Write a Python function to find the maximum value in a list of tuples.",
        "test": "assert find_Max_Tuples([(1,2),(3,4)]) == (3,4)\nassert find_Max_Tuples([(5,6,7),(1,2)]) == (5,6,7)",
        "fn_name": "find_Max_Tuples",
    },
    {
        "task_id": "MBPP/17",
        "prompt": "Write a Python function to check whether the given string is a valid IP address (IPv4).",
        "test": "assert is_valid_ip('192.168.0.1') == True\nassert is_valid_ip('256.0.0.1') == False\nassert is_valid_ip('abc') == False\nassert is_valid_ip('0.0.0.0') == True",
        "fn_name": "is_valid_ip",
    },
    {
        "task_id": "MBPP/18",
        "prompt": "Write a Python function to count the number of prime numbers less than a given non-negative number.",
        "test": "assert count_Primes_nums(5) == 2\nassert count_Primes_nums(10) == 4\nassert count_Primes_nums(0) == 0",
        "fn_name": "count_Primes_nums",
    },
    {
        "task_id": "MBPP/20",
        "prompt": "Write a Python function to find the maximum and minimum number from a list and return them as a tuple (min, max).",
        "test": "assert max_min([1,2,3,4,5]) == (1,5)\nassert max_min([5]) == (5,5)",
        "fn_name": "max_min",
    },
    {
        "task_id": "MBPP/22",
        "prompt": "Write a Python function that takes a list of integers and returns a new list with duplicates removed and elements in the original order.",
        "test": "assert remove_duplic_list([1,2,1,3,2]) == [1,2,3]\nassert remove_duplic_list([]) == []",
        "fn_name": "remove_duplic_list",
    },
    {
        "task_id": "MBPP/26",
        "prompt": "Write a Python function to find the product of all elements in a list.",
        "test": "assert multiply_list([1,2,3,4]) == 24\nassert multiply_list([3,2,2,1]) == 12\nassert multiply_list([1]) == 1",
        "fn_name": "multiply_list",
    },
    {
        "task_id": "MBPP/27",
        "prompt": "Write a Python function to count the number of words in a given string.",
        "test": "assert count_words('hello world foo') == 3\nassert count_words('') == 0\nassert count_words('one') == 1",
        "fn_name": "count_words",
    },
    {
        "task_id": "MBPP/28",
        "prompt": "Write a Python function to convert a list of characters into a string.",
        "test": "assert convert_list_char(['p','y','t','h','o','n']) == 'python'\nassert convert_list_char([]) == ''",
        "fn_name": "convert_list_char",
    },
    {
        "task_id": "MBPP/29",
        "prompt": "Write a Python function to find the maximum length of a consecutive sequence in a list of integers.",
        "test": "assert max_consecutive_sequence([1,2,3,5,6,7,8]) == 4\nassert max_consecutive_sequence([1]) == 1\nassert max_consecutive_sequence([]) == 0",
        "fn_name": "max_consecutive_sequence",
    },
    {
        "task_id": "MBPP/31",
        "prompt": "Write a Python function to flatten a nested list of any depth into a single list.",
        "test": "assert flatten([1,[2,[3,[4]]],5]) == [1,2,3,4,5]\nassert flatten([1,2,3]) == [1,2,3]\nassert flatten([]) == []",
        "fn_name": "flatten",
    },
    {
        "task_id": "MBPP/35",
        "prompt": "Write a Python function to find the number of ways to make change for a given amount using coins of given denominations.",
        "test": "assert count_ways(4,[1,2,3]) == 4\nassert count_ways(0,[1,2]) == 1",
        "fn_name": "count_ways",
    },
    {
        "task_id": "MBPP/36",
        "prompt": "Write a Python function that takes a dictionary and returns a new dictionary with keys and values swapped.",
        "test": "assert swap_dict({'a':1,'b':2}) == {1:'a',2:'b'}\nassert swap_dict({}) == {}",
        "fn_name": "swap_dict",
    },
    {
        "task_id": "MBPP/39",
        "prompt": "Write a Python function to check whether all elements in a list are distinct.",
        "test": "assert all_distinct([1,2,3]) == True\nassert all_distinct([1,2,1]) == False\nassert all_distinct([]) == True",
        "fn_name": "all_distinct",
    },
    {
        "task_id": "MBPP/40",
        "prompt": "Write a Python function to return the nth Fibonacci number (0-indexed, fib(0)=0, fib(1)=1).",
        "test": "assert fib(0)==0\nassert fib(1)==1\nassert fib(7)==13\nassert fib(10)==55",
        "fn_name": "fib",
    },
]

# ─── SUMMARY ACCURACY SCENARIOS (5) ────────────────────────────────────────────

SUMMARY_SCENARIOS = [
    {
        "id": "summary/pr_breaking",
        "label": "PR with Breaking API Change",
        "content": """
PULL REQUEST #847 — Merge: refactor-auth-v2 → main
Author: @mlee | Reviewer: @jsmith | Status: APPROVED

## Summary
Migrates authentication from session cookies to JWT tokens. Removes legacy
`/api/v1/auth/session` endpoint entirely. Adds new `/api/v2/auth/token` endpoint.

## Files Changed
- `src/auth/middleware.py` (+312, -89)
- `src/auth/session.py` (DELETED)
- `src/auth/jwt_handler.py` (NEW, +201)
- `src/api/routes.py` (+12, -8)
- `src/api/v1/auth.py` (+3, -47) — endpoint removed, returns 410 Gone
- `tests/test_auth.py` (+156, -72)

## Breaking Changes
1. `GET /api/v1/auth/session` is REMOVED (returns 410). All clients must migrate.
2. `AuthMiddleware.__init__()` no longer accepts `session_store` parameter.
3. Token expiry changed from 7 days to 24 hours. Refresh tokens now required.
4. `request.user.session_id` attribute no longer exists. Use `request.user.token_jti`.

## Bug Fixes (Silent)
- Fixed: logout did not invalidate server-side session, allowing reuse after logout.
  CVE-2024-9182 — CVSS 7.4 (High). Fixed by removing session store entirely.

## Migration Guide
Clients must:
1. Call POST /api/v2/auth/token with {username, password} to get {access_token, refresh_token}
2. Send `Authorization: Bearer <access_token>` header instead of session cookie
3. Call POST /api/v2/auth/refresh with {refresh_token} before access_token expires (24h)

## Test Coverage: 94% (+8%)
""",
        "keywords": [
            "breaking", "/api/v1/auth/session", "removed", "JWT", "24 hours",
            "CVE-2024-9182", "logout", "session_id", "token_jti", "refresh",
            "410", "AuthMiddleware"
        ],
    },
    {
        "id": "summary/crash_stacktrace",
        "label": "Production Crash Stack Trace",
        "content": """
INCIDENT: INC-2847 | Severity: P1 | Started: 2026-03-12 14:23 UTC

Service: payment-processor | Environment: production-eu-west-1
Error rate: 100% (all payment requests failing) | Duration: 47 minutes

STACK TRACE (repeated, all workers):
  File "/app/payments/processor.py", line 287, in charge_card
    result = self.gateway.authorize(amount=amount, currency=currency,
  File "/app/payments/gateway.py", line 156, in authorize
    response = self._client.post(self.endpoint, json=payload, timeout=self.timeout)
  File "/app/lib/http_client.py", line 89, in post
    raise ConnectionError(f"SSL handshake failed: {e}")
ConnectionError: SSL handshake failed: [SSL: CERTIFICATE_VERIFY_FAILED]
  certificate verify failed: certificate has expired (_ssl.c:1123)

ROOT CAUSE:
  Stripe gateway SSL certificate expired at 14:22 UTC. Wildcard cert
  *.stripe-payments.com expired 2026-03-12. Our http_client.py enforces
  verify=True with no fallback. The cert was on Stripe's CDN, not our infra.

RESOLUTION:
  Stripe pushed renewed cert at 15:10 UTC. All services recovered automatically.
  No data loss. No charges processed during outage (gateway rejected all requests
  before any charges were submitted).

CONTRIBUTING FACTOR:
  payment-processor has no circuit breaker. All 847 requests during outage
  received 500 errors instead of a graceful queue/retry. Ticket ENG-4421
  created to add circuit breaker pattern.

AFFECTED SERVICES: payment-processor, invoice-service (depends on it)
NOT AFFECTED: refund-processor (uses different gateway endpoint)
""",
        "keywords": [
            "SSL", "certificate expired", "14:22", "Stripe", "*.stripe-payments.com",
            "no data loss", "847 requests", "circuit breaker", "ENG-4421",
            "invoice-service", "payment-processor", "15:10"
        ],
    },
    {
        "id": "summary/security_patch",
        "label": "Security Patch with CVE",
        "content": """
SECURITY ADVISORY — INTERNAL
Classification: CONFIDENTIAL | Do not share externally until public disclosure

Vulnerability: SQL Injection in User Search API
CVE: CVE-2026-1147 | CVSS Score: 9.1 (Critical)
Affected Versions: api-server 2.1.0 through 2.4.7
Fixed In: api-server 2.4.8 (released today, forced rollout in progress)
Patched File: src/api/users/search.py, line 142

DESCRIPTION:
  The `GET /api/users/search?q=` endpoint passes the `q` parameter directly
  to a SQLAlchemy `text()` query without parameterization. An attacker can
  exfiltrate any table in the database using UNION-based injection.

PROOF OF CONCEPT (DO NOT DISTRIBUTE):
  /api/users/search?q=' UNION SELECT username,password,NULL FROM admin_users--

EXPLOITABILITY:
  - Authentication required (any valid API key)
  - Affects all database backends (PostgreSQL, MySQL)
  - API key rotation does NOT mitigate — all existing keys are vulnerable

IMMEDIATE ACTIONS REQUIRED:
  1. All instances must upgrade to 2.4.8 within 4 hours (SLA: by 18:00 UTC)
  2. Rotate ALL API keys after upgrade (old keys may have been logged)
  3. Audit database logs from 2026-01-01 for suspicious UNION queries
  4. Enable WAF rule SQLI-007 as defense-in-depth

WORKAROUND (if upgrade impossible): Set SEARCH_DISABLED=1 env var to return 503.
""",
        "keywords": [
            "CVE-2026-1147", "CVSS 9.1", "SQL injection", "/api/users/search",
            "2.4.8", "versions 2.1.0 through 2.4.7", "API key rotation",
            "UNION", "18:00 UTC", "WAF rule SQLI-007", "SEARCH_DISABLED",
            "audit logs 2026-01-01"
        ],
    },
    {
        "id": "summary/refactor_interfaces",
        "label": "Refactor Changing 3 Module Interfaces",
        "content": """
REFACTOR SUMMARY — ENG-3891: DataPipeline v3 Interface Migration

This refactor changes three core module interfaces as part of the DataPipeline v3
redesign. All changes are breaking for downstream consumers.

## Module 1: DataReader (src/pipeline/reader.py)
OLD: DataReader(path: str, format: str = 'csv')
NEW: DataReader(source: DataSource, config: ReaderConfig)
CHANGE: Constructor no longer accepts raw file paths. Must use DataSource.from_path().
IMPACT: 23 call sites across 8 files need updating.
NEW DEPENDENCY: DataSource object must be constructed and passed in.

## Module 2: Transformer (src/pipeline/transformer.py)
OLD: transformer.apply(data: list) -> list
NEW: transformer.apply(data: DataBatch) -> DataBatch
CHANGE: Input/output types changed from raw lists to DataBatch objects.
IMPACT: Cannot mix old and new pipeline stages. All stages must be migrated together.
PERFORMANCE: DataBatch uses zero-copy slicing; expect 30-40% memory reduction.

## Module 3: DataWriter (src/pipeline/writer.py)
OLD: writer.flush() returns None, raises on error
NEW: writer.flush() returns WriteResult(success: bool, rows: int, errors: list)
CHANGE: Errors no longer raise exceptions; check WriteResult.success instead.
IMPACT: All existing try/except blocks around flush() will silently swallow errors.
        Must add: if not result.success: handle_errors(result.errors)

## Migration Deadline: 2026-04-15
All services must complete migration. After deadline, v2 interfaces will be
removed and old consumers will fail at import time.
""",
        "keywords": [
            "DataReader", "DataSource.from_path()", "23 call sites",
            "DataBatch", "Transformer", "30-40% memory",
            "WriteResult", "flush()", "silently swallow",
            "2026-04-15", "DataWriter", "zero-copy"
        ],
    },
    {
        "id": "summary/perf_regression",
        "label": "Performance Regression Report",
        "content": """
PERFORMANCE REGRESSION REPORT — Week of 2026-05-06
Auto-generated by perfbot | Threshold: >20% regression triggers alert

REGRESSION DETECTED: POST /api/reports/generate
  Baseline (p50): 340ms | Current (p50): 1,847ms | Change: +443%
  Baseline (p99): 2.1s  | Current (p99): 18.4s   | Change: +776%
  First detected: 2026-05-07 03:14 UTC
  Correlated deploy: v4.7.2 at 2026-05-07 02:58 UTC

ROOT CAUSE ANALYSIS:
  PR #892 (merged in v4.7.2) added eager loading to ReportGenerator.
  The change loads ALL related dataset records upfront via:
    datasets = Dataset.objects.prefetch_related('rows', 'columns', 'metadata')
  For large reports (>10K rows), this loads millions of ORM objects into memory.
  Average report has 47K rows. Memory per request: 1.2 GB (was 12 MB).
  Worker processes hit OOM → kernel swap → extreme latency.

AFFECTED ENDPOINTS:
  - POST /api/reports/generate (443% slower) — PRIMARY
  - GET /api/reports/{id}/export (87% slower) — uses same generator
  - GET /api/dashboard/summary (12% slower) — shares DB connection pool

NOT AFFECTED: All other endpoints. No data corruption observed.

FIX:
  Revert to lazy loading for rows/columns. Keep eager load only for metadata.
  Fix is in PR #901 (pending review). ETA: deployed by 2026-05-09 18:00 UTC.

IMMEDIATE MITIGATION:
  Rate limit /api/reports/generate to 2 concurrent requests per tenant.
  Applied via nginx config at 2026-05-08 09:30 UTC. Reduces OOM rate by 80%.
""",
        "keywords": [
            "POST /api/reports/generate", "+443%", "v4.7.2", "PR #892",
            "eager loading", "47K rows", "1.2 GB", "PR #901",
            "GET /api/reports/{id}/export", "2026-05-09 18:00 UTC",
            "rate limit 2 concurrent", "nginx"
        ],
    },
]

# ─── INFO PRESERVATION SCENARIOS (5) ───────────────────────────────────────────

INFO_SCENARIOS = [
    {
        "id": "info/config_module",
        "label": "Config Module Facts",
        "content": """
# config.py — Application Configuration Loader

class Config:
    DEFAULT_TIMEOUT = 30          # seconds
    MAX_RETRIES = 3
    CACHE_TTL = 3600              # 1 hour
    MAX_CONNECTIONS = 100
    DB_POOL_SIZE = 10
    SECRET_KEY_ENV = 'APP_SECRET_KEY'  # must be 256-bit (32 bytes) minimum
    LOG_LEVEL = 'WARNING'         # production default
    FEATURE_FLAGS_ENDPOINT = 'http://flags.internal:8090/v2'
    METRICS_PORT = 9090
    HEALTHCHECK_PATH = '/healthz'

    def __init__(self, env='production'):
        self.env = env
        if env == 'development':
            self.LOG_LEVEL = 'DEBUG'
            self.MAX_CONNECTIONS = 10
            self.CACHE_TTL = 60      # 1 minute in dev
        elif env == 'test':
            self.CACHE_TTL = 0       # no caching in tests
            self.MAX_RETRIES = 0     # fail fast in tests

    @classmethod
    def from_env(cls):
        env = os.environ.get('APP_ENV', 'production')
        return cls(env=env)
""",
        "questions": [
            ("What is the default timeout?", "30"),
            ("What is the cache TTL in production?", "3600"),
            ("What port does metrics run on?", "9090"),
            ("What is the healthcheck path?", "/healthz"),
            ("What is the max DB pool size?", "10"),
            ("What log level is used in production?", "WARNING"),
            ("What is CACHE_TTL in test environment?", "0"),
            ("What is the minimum size for SECRET_KEY?", "32 bytes"),
        ],
    },
    {
        "id": "info/api_schema",
        "label": "API Schema Facts",
        "content": """
# User API — Field Definitions and Validation Rules

POST /api/v3/users
Required fields:
  - email: string, must be valid email, max 254 chars, unique across users
  - username: string, 3-32 chars, alphanumeric + underscore only, case-insensitive unique
  - password: string, min 12 chars, must contain uppercase, lowercase, digit, special char

Optional fields:
  - display_name: string, max 64 chars, default = username
  - timezone: string, IANA timezone, default = 'UTC'
  - role: enum ['user', 'admin', 'readonly'], default = 'user'
  - invite_code: string, if provided must match an active invite, consumed on use

Rate limiting: 5 requests per IP per minute
Response: 201 Created with {id, email, username, created_at}
Errors: 400 (validation), 409 (email/username conflict), 429 (rate limit)

PUT /api/v3/users/{id}
Auth: Bearer token, must be same user OR admin role
Updatable: display_name, timezone, password
NOT updatable: email, username, role (use admin API for role changes)
Rate limiting: 10 requests per user per minute
""",
        "questions": [
            ("What is the minimum password length?", "12"),
            ("What is the max username length?", "32"),
            ("What is the default role for new users?", "user"),
            ("What HTTP status is returned on conflict?", "409"),
            ("What is the default timezone?", "UTC"),
            ("Can you update email via PUT /api/v3/users/{id}?", "no"),
            ("What is the rate limit for POST /api/v3/users?", "5 per minute"),
            ("What is the max display_name length?", "64"),
        ],
    },
    {
        "id": "info/deployment",
        "label": "Deployment Facts",
        "content": """
# Deployment Runbook — api-server v2.5.0

Release Date: 2026-05-10
Deployment Window: Saturday 02:00-04:00 UTC (maintenance window)
Expected Downtime: ~3 minutes during DB migration
Rollback Time Estimate: 8 minutes

Pre-deployment Checklist:
1. Backup production DB (automated at 01:00 UTC via pg_dump to s3://backups/prod)
2. Verify s3://backups/prod/20260510_0100.dump.gz exists and is > 500MB
3. Drain load balancer (set weight=0 on eu-west-1a, wait 30s, then eu-west-1b)
4. Set MAINTENANCE_MODE=1 on all app servers

DB Migration (migration_047_add_audit_table.py):
- Adds audit_log table (no existing data affected)
- Adds index on users.last_login (concurrent index, no lock)
- Estimated duration: 90 seconds on production data size

Environment Variables CHANGED in v2.5.0:
  - LEGACY_API_COMPAT removed (breaking: remove from all environments)
  - AUDIT_LOG_ENABLED added, default=false, set to true in production only

Ports: app remains on 8080, new admin endpoint on 8081 (firewall rule needed)
""",
        "questions": [
            ("What time is the deployment window?", "02:00"),
            ("How long is expected downtime?", "3 minutes"),
            ("What port is the new admin endpoint on?", "8081"),
            ("Is LEGACY_API_COMPAT being added or removed?", "removed"),
            ("What is the rollback time estimate?", "8 minutes"),
            ("Where are DB backups stored?", "s3://backups/prod"),
            ("What does migration 047 add?", "audit_log"),
            ("What is the default for AUDIT_LOG_ENABLED?", "false"),
        ],
    },
    {
        "id": "info/error_handling",
        "label": "Error Handling Facts",
        "content": """
# Error Handling Strategy — payment-service

## Retry Policy
- Network errors (ConnectionError, TimeoutError): retry 3 times, exponential backoff
  starting at 1s (1s, 2s, 4s). Total max wait: 7 seconds.
- HTTP 429 (rate limit): retry with Retry-After header value, max 2 retries.
- HTTP 503 (service unavailable): retry 2 times with 5s fixed delay.
- HTTP 402 (payment required / declined): DO NOT retry. Log and return immediately.
- HTTP 500: retry once after 2s. If still 500, alert on-call (PagerDuty P2).

## Dead Letter Queue
- Messages that fail all retries go to SQS queue: payment-dlq-prod
- DLQ retention: 14 days
- DLQ alerts: if depth > 100, PagerDuty P1

## Circuit Breaker (per downstream service)
- Opens after 5 consecutive failures within 60 seconds
- Half-open after 30 seconds (allows 1 probe request)
- Closes after 3 consecutive successes in half-open state
- Stripe gateway: circuit breaker DISABLED (pending ENG-4421)
""",
        "questions": [
            ("How many retries for network errors?", "3"),
            ("What is the max total wait time for network retries?", "7 seconds"),
            ("Should HTTP 402 be retried?", "no"),
            ("What is the DLQ retention period?", "14 days"),
            ("What depth triggers a P1 alert?", "100"),
            ("How long before circuit breaker half-opens?", "30 seconds"),
            ("After how many failures does circuit breaker open?", "5"),
            ("Is Stripe gateway circuit breaker enabled?", "no"),
        ],
    },
    {
        "id": "info/service_limits",
        "label": "Service Limits Facts",
        "content": """
# Service Limits — data-export-api (as of v3.1.0)

## Request Limits
- Max file size per export: 5 GB
- Max rows per export: 10,000,000
- Max concurrent exports per tenant: 3
- Export timeout: 30 minutes (exports exceeding this are cancelled, not saved)
- Max columns per export: 500

## Rate Limits (per API key)
- Export creation: 20 per hour, 200 per day
- Status checks: 120 per minute
- Download requests: 50 per hour per file

## Storage
- Exports retained for: 72 hours after completion
- Failed exports retained for: 24 hours (for debugging)
- Storage backend: S3 (us-east-1 primary, eu-west-1 replica)
- Encryption: AES-256 at rest, TLS 1.3 in transit

## File Formats Supported
- CSV, JSON, Parquet, XLSX (max 1M rows for XLSX due to Excel limit)
- Compression: gzip, zstd (default: gzip)
""",
        "questions": [
            ("What is the max file size per export?", "5 GB"),
            ("What is the max concurrent exports per tenant?", "3"),
            ("How long are exports retained after completion?", "72 hours"),
            ("What is the export timeout?", "30 minutes"),
            ("What is the max rows for XLSX format?", "1 million"),
            ("What is the default compression?", "gzip"),
            ("What is the max export creation rate per day?", "200"),
            ("What encryption is used at rest?", "AES-256"),
        ],
    },
]

# ─── QUIXBUGS-STYLE BUG FIX SCENARIOS (5) ─────────────────────────────────────
# Source benchmark: QuixBugs, a Python/Java program-repair benchmark based on
# one-line bugs from the Quixey Challenge. These compact tasks evaluate the
# local coder role: read a bug report plus failing tests, then return a patch.

BUGFIX_SCENARIOS = [
    {
        "task_id": "QB/bitcount",
        "prompt": """def bitcount(n):
    count = 0
    while n:
        n ^= n - 1
        count += 1
    return count
""",
        "description": "Fix the function so it returns the number of 1 bits in a non-negative integer.",
        "test": "assert bitcount(127) == 7\nassert bitcount(128) == 1\nassert bitcount(3005) == 9\nassert bitcount(0) == 0",
    },
    {
        "task_id": "QB/gcd",
        "prompt": """def gcd(a, b):
    if b == 0:
        return a
    return gcd(a % b, a)
""",
        "description": "Fix the recursive Euclidean algorithm.",
        "test": "assert gcd(12, 8) == 4\nassert gcd(25, 15) == 5\nassert gcd(0, 5) == 5\nassert gcd(17, 13) == 1",
    },
    {
        "task_id": "QB/parentheses",
        "prompt": """def is_valid_parenthesization(parens):
    depth = 0
    for paren in parens:
        if paren == '(':
            depth += 1
        else:
            depth -= 1
    return depth == 0
""",
        "description": "Fix the validator so it rejects strings that close before they open.",
        "test": "assert is_valid_parenthesization('()') is True\nassert is_valid_parenthesization('(())()') is True\nassert is_valid_parenthesization(')(') is False\nassert is_valid_parenthesization('(()') is False",
    },
    {
        "task_id": "QB/find_in_sorted",
        "prompt": """def find_in_sorted(arr, x):
    def binsearch(start, end):
        if start == end:
            return -1
        mid = start + (end - start) // 2
        if x < arr[mid]:
            return binsearch(start, mid)
        elif x > arr[mid]:
            return binsearch(mid, end)
        else:
            return mid
    return binsearch(0, len(arr))
""",
        "description": "Fix the binary search so all sorted-list edge cases terminate correctly.",
        "test": "assert find_in_sorted([1, 2, 3, 4, 5], 1) == 0\nassert find_in_sorted([1, 2, 3, 4, 5], 5) == 4\nassert find_in_sorted([1, 2, 3, 4, 5], 6) == -1\nassert find_in_sorted([], 1) == -1",
    },
    {
        "task_id": "QB/max_sublist_sum",
        "prompt": """def max_sublist_sum(arr):
    max_ending_here = 0
    max_so_far = 0
    for x in arr:
        max_ending_here = max(0, max_ending_here + x)
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far
""",
        "description": "Fix the maximum subarray sum so all-negative lists return the largest element.",
        "test": "assert max_sublist_sum([4, -1, 2, 1]) == 6\nassert max_sublist_sum([-2, -3, -1, -5]) == -1\nassert max_sublist_sum([5]) == 5\nassert max_sublist_sum([]) == 0",
    },
]

# ─── LM STUDIO BACKEND ─────────────────────────────────────────────────────────
#
# Two layers:
#   * `lms` CLI for lifecycle (server start/stop, load/unload models, listing).
#   * REST API (OpenAI-compatible) for the actual inference calls.
#
# We keep state about which model the current process has loaded in
# `CURRENT_MODEL`, so the llm() / measure_tps() calls pass the right id.

CURRENT_MODEL: Optional[str] = None
CURRENT_API_MODEL: Optional[str] = None


def _lms_bin() -> str:
    """Locate the `lms` CLI or raise a helpful error."""
    bin_path = shutil.which("lms")
    if bin_path:
        return bin_path
    # Common macOS install locations, depending on LM Studio version.
    fallbacks = [
        Path.home() / ".lmstudio" / "bin" / "lms",
        Path.home() / ".cache" / "lm-studio" / "bin" / "lms",
    ]
    for fallback in fallbacks:
        if fallback.exists():
            return str(fallback)
    raise RuntimeError(
        "Could not find the `lms` CLI on $PATH. Open LM Studio once, then run "
        "`lms bootstrap` so the CLI is installed."
    )


def _run_lms(args: List[str], *, check: bool = False, timeout: Optional[int] = 60,
             log_path: Optional[Path] = None) -> subprocess.CompletedProcess:
    """Run an `lms` subcommand. Optionally tee output to a log file."""
    cmd = [_lms_bin(), *args]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(f"\n--- lms {' '.join(args)} @ {datetime.now().isoformat()} (rc={proc.returncode}) ---\n")
            if proc.stdout:
                f.write(proc.stdout)
            if proc.stderr:
                f.write("\n[stderr]\n")
                f.write(proc.stderr)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"lms {' '.join(args)} failed (rc={proc.returncode}): "
            f"{(proc.stderr or proc.stdout or '').strip()[:400]}"
        )
    return proc


def _server_alive() -> bool:
    try:
        urllib.request.urlopen(f"{LM_STUDIO_URL}/v1/models", timeout=2)
        return True
    except Exception:
        return False


def ensure_lm_studio_server() -> None:
    """Make sure LM Studio's OpenAI server is reachable."""
    if _server_alive():
        return
    print(f"  [LM Studio] Server not reachable at {LM_STUDIO_URL}, starting via lms ...")
    _run_lms(["server", "start"], check=False)
    deadline = time.time() + SERVER_TIMEOUT
    while time.time() < deadline:
        if _server_alive():
            print(f"  [LM Studio] Server up at {LM_STUDIO_URL}")
            return
        time.sleep(2)
    raise RuntimeError(
        f"LM Studio server did not become reachable at {LM_STUDIO_URL} within {SERVER_TIMEOUT}s. "
        "Open LM Studio and toggle the local server on, then retry."
    )


def discover_lm_studio_models() -> List[Dict]:
    """Return one entry per loadable LLM in LM Studio.

    LM Studio exposes variant ids such as `model@q8_0`, but current `lms load`
    versions only accept the base model key for grouped models. Benchmark the
    selected variant through the loadable base key and record the available
    variants for traceability.
    """
    proc = _run_lms(["ls", "--llm", "--json"], check=False, timeout=30)
    raw = (proc.stdout or "").strip()
    parsed = []
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                # Some versions wrap the list under "models" or similar.
                for key in ("models", "items", "data"):
                    if isinstance(data.get(key), list):
                        data = data[key]
                        break
            if isinstance(data, list):
                parsed = data
        except json.JSONDecodeError:
            parsed = []
    # Fallback: plain `lms ls` output if JSON failed or returned nothing.
    if not parsed:
        plain = _run_lms(["ls", "--llm"], check=False, timeout=30).stdout or ""
        for line in plain.splitlines():
            line = line.strip()
            if not line or line.startswith(("PARAMETER", "LLM", "—", "-")):
                continue
            parts = line.split()
            if parts:
                parsed.append({"modelKey": parts[0]})

    models = []
    rank = 0
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        etype = (entry.get("type") or "").lower()
        if etype and etype not in {"llm", "vlm"}:
            # Skip embeddings, etc.
            continue

        base_key = (entry.get("modelKey") or entry.get("model_key")
                    or entry.get("identifier") or entry.get("path") or entry.get("name"))
        if not base_key:
            continue

        base_label = entry.get("displayName") or entry.get("display_name") or base_key
        publisher = entry.get("publisher") or ""
        model_path = entry.get("path") or entry.get("indexedModelIdentifier") or base_key
        architecture = entry.get("architecture") or entry.get("type") or "llm"
        fmt = entry.get("format") or ""
        size_bytes = entry.get("sizeBytes") or entry.get("size") or 0
        params = entry.get("paramsString") or ""
        max_context = entry.get("maxContextLength") or 0
        quant_obj = entry.get("quantization")
        default_quant = quant_obj.get("name") if isinstance(quant_obj, dict) else (quant_obj or "")

        variants = entry.get("variants")
        variant_ids = [v for v in variants if isinstance(v, str)] if isinstance(variants, list) else []
        selected_variant = entry.get("selectedVariant") or (variant_ids[0] if variant_ids else "")
        quant = default_quant or ""
        if selected_variant and "@" in selected_variant and not quant:
            quant = selected_variant.rsplit("@", 1)[-1]

        rank += 1
        label = f"{base_label} ({quant})" if quant else base_label
        models.append({
            "rank": rank,
            "id": base_key,
            "base_key": base_key,
            "label": label,
            "publisher": publisher,
            "path": model_path,
            "type": architecture,
            "format": fmt,
            "size_bytes": size_bytes,
            "params": params,
            "max_context": max_context,
            "quantization": quant,
            "selected_variant": selected_variant,
            "available_variants": variant_ids,
        })
    return models


def load_model(model_key: str, log_path: Optional[Path] = None) -> None:
    """Load a model into LM Studio and wait until it answers /v1/models."""
    global CURRENT_MODEL, CURRENT_API_MODEL
    print(f"  [LM Studio] Loading {model_key} ...")
    # Unload anything already resident so memory pressure stays predictable.
    _run_lms(["unload", "--all"], check=False, timeout=120, log_path=log_path)
    t0 = time.time()
    proc = _run_lms(["load", model_key, "-y"], check=False, timeout=LOAD_TIMEOUT, log_path=log_path)
    if proc.returncode != 0:
        raise RuntimeError(
            f"lms load {model_key!r} failed (rc={proc.returncode}): "
            f"{(proc.stderr or proc.stdout or '').strip()[:400]}"
        )
    # Confirm it shows up in /v1/models.
    deadline = time.time() + LOAD_TIMEOUT
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{LM_STUDIO_URL}/v1/models", timeout=5) as resp:
                listed = json.loads(resp.read()).get("data", [])
            ids = [m.get("id", "") for m in listed]
            if ids:
                api_id = next(
                    (mid for mid in ids
                     if model_key == mid or model_key.endswith(mid) or mid.endswith(model_key)),
                    ids[0],
                )
                CURRENT_MODEL = model_key
                CURRENT_API_MODEL = api_id
                print(f"  [LM Studio] Loaded in {time.time() - t0:.1f}s (api id={api_id})")
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"Model {model_key!r} did not appear in /v1/models within {LOAD_TIMEOUT}s")


def unload_model(model_key: Optional[str] = None, log_path: Optional[Path] = None) -> None:
    """Unload the named model (or everything if none given)."""
    global CURRENT_MODEL, CURRENT_API_MODEL
    target = model_key or CURRENT_MODEL
    if target:
        print(f"  [LM Studio] Unloading {target} ...")
        _run_lms(["unload", target], check=False, timeout=120, log_path=log_path)
    else:
        _run_lms(["unload", "--all"], check=False, timeout=120, log_path=log_path)
    CURRENT_MODEL = None
    CURRENT_API_MODEL = None
    time.sleep(1)


def _api_model_id() -> str:
    """The id to pass to /v1/chat/completions for the currently-loaded model."""
    if not CURRENT_API_MODEL:
        raise RuntimeError("No model is loaded.")
    return CURRENT_API_MODEL


# ─── LLM CALL ──────────────────────────────────────────────────────────────────

def llm(messages: list, max_tokens: int = 1024, temperature: float = 0.1) -> str:
    payload = json.dumps({
        "model": _api_model_id(),
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{LM_STUDIO_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


# ─── CODE EXECUTION ─────────────────────────────────────────────────────────────

def extract_code(response: str) -> str:
    """Extract Python code block from model response."""
    # Try ```python ... ``` block first
    m = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try ``` ... ``` block
    m = re.search(r"```\s*(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fall back to raw response
    return response.strip()


def function_name_from_prompt(prompt: str) -> str:
    m = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", prompt)
    if not m:
        raise ValueError("Could not find function name in prompt")
    return m.group(1)


def extract_named_function(code: str, fn_name: str) -> str:
    """Return just the requested function when the model includes extra text."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            segment = ast.get_source_segment(code, node)
            return segment.strip() if segment else code
    return code


def build_function_from_body(prompt: str, body: str) -> str:
    body = body.strip()
    if not body:
        body = "pass"
    return prompt.rstrip() + "\n" + indent(body, "    ")


def prepare_function_code(raw: str, fn_name: str, prompt: str = None) -> str:
    code = extract_code(raw)
    code = re.sub(r"^(?:Here(?:'s| is).*?:|Corrected code:)\s*", "", code, flags=re.IGNORECASE).strip()
    if f"def {fn_name}" in code:
        return extract_named_function(code, fn_name)
    if prompt is not None:
        return build_function_from_body(prompt, code)
    return code


def run_code_test(code: str, test: str, timeout: int = 10) -> tuple:
    """Run code + test assertions in isolated subprocess."""
    full = f"{code}\n\n{test}"
    try:
        result = subprocess.run(
            [sys.executable, "-c", full],
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        err = (result.stderr or result.stdout or "").strip()
        return result.returncode == 0, err[:300]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)[:300]


def normalized_text(value: str) -> str:
    value = value.casefold()
    value = value.replace("—", "-").replace("–", "-")
    value = re.sub(r"[\s`'\"_*]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def contains_fact(text: str, fact: str) -> bool:
    text_norm = normalized_text(text)
    fact_norm = normalized_text(fact)
    if fact_norm in text_norm:
        return True
    compact_text = re.sub(r"[^a-z0-9]+", "", text_norm)
    compact_fact = re.sub(r"[^a-z0-9]+", "", fact_norm)
    return bool(compact_fact and compact_fact in compact_text)


# ─── BENCHMARK FUNCTIONS ────────────────────────────────────────────────────────

def bench_humaneval() -> dict:
    passed = 0
    details = []
    for prob in HUMANEVAL:
        fn_name = function_name_from_prompt(prob["prompt"])
        prompt = (
            f"Complete this HumanEval Python task. Return ONLY the complete function "
            f"definition for `{fn_name}` with no markdown, no prose, and no tests.\n\n{prob['prompt']}"
        )
        try:
            raw = llm([{"role": "user", "content": prompt}], max_tokens=512)
            code = prepare_function_code(raw, fn_name, prob["prompt"])
            ok, err = run_code_test(code, prob["test"])
        except Exception as e:
            ok = False
            err = str(e)[:300]
        passed += int(ok)
        detail = {"task": prob["task_id"], "pass": ok}
        if not ok and err:
            detail["error"] = err
        details.append(detail)
        print(f"    HumanEval {prob['task_id']}: {'✓' if ok else '✗'}")
    score = passed / len(HUMANEVAL)
    return {"score": score, "passed": passed, "total": len(HUMANEVAL), "details": details}


def bench_mbpp() -> dict:
    passed = 0
    details = []
    for prob in MBPP:
        prompt = (
            f"Write a Python function named `{prob['fn_name']}` that does the following:\n"
            f"{prob['prompt']}\n\n"
            f"Return ONLY the function definition, no test code."
        )
        try:
            raw = llm([{"role": "user", "content": prompt}], max_tokens=512)
            code = prepare_function_code(raw, prob["fn_name"])
            ok, err = run_code_test(code, prob["test"])
        except Exception as e:
            ok = False
            err = str(e)[:300]
        passed += int(ok)
        detail = {"task": prob["task_id"], "pass": ok}
        if not ok and err:
            detail["error"] = err
        details.append(detail)
        print(f"    MBPP {prob['task_id']}: {'✓' if ok else '✗'}")
    score = passed / len(MBPP)
    return {"score": score, "passed": passed, "total": len(MBPP), "details": details}


def bench_bugfix() -> dict:
    passed = 0
    details = []
    for prob in BUGFIX_SCENARIOS:
        fn_name = function_name_from_prompt(prob["prompt"])
        prompt = (
            "You are a local coding model receiving a precise bug-fix task from an orchestrator.\n"
            f"Task: {prob['description']}\n\n"
            "Buggy code:\n"
            f"{prob['prompt']}\n"
            "Return ONLY the complete corrected Python function. No markdown, no prose, no tests."
        )
        try:
            raw = llm([{"role": "user", "content": prompt}], max_tokens=512)
            code = prepare_function_code(raw, fn_name)
            ok, err = run_code_test(code, prob["test"])
        except Exception as e:
            ok = False
            err = str(e)[:300]
        passed += int(ok)
        detail = {"task": prob["task_id"], "pass": ok}
        if not ok and err:
            detail["error"] = err
        details.append(detail)
        print(f"    BugFix {prob['task_id']}: {'✓' if ok else '✗'}")
    score = passed / len(BUGFIX_SCENARIOS)
    return {"score": score, "passed": passed, "total": len(BUGFIX_SCENARIOS), "details": details}


def bench_summary() -> dict:
    SUMMARY_PROMPT = (
        "You are a senior engineer summarizing this for an AI orchestrator. "
        "Write a summary under 200 words that preserves ALL breaking changes, bugs, "
        "security issues, deadlines, endpoints, identifiers, PR numbers, versions, and critical numbers. "
        "Use bullet points. Keep exact technical values verbatim when possible. "
        "Do not omit anything that would block follow-up coding work."
    )
    total_score = 0.0
    details = []
    for scenario in SUMMARY_SCENARIOS:
        try:
            summary = llm([
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": scenario["content"]},
            ], max_tokens=300)
            hits = sum(1 for kw in scenario["keywords"] if contains_fact(summary, kw))
            score = hits / len(scenario["keywords"])
        except Exception:
            score = 0.0
            hits = 0
            summary = ""
        total_score += score
        details.append({
            "scenario": scenario["id"],
            "keywords_hit": hits,
            "keywords_total": len(scenario["keywords"]),
            "score": round(score, 3),
            "preview": summary[:240],
        })
        print(f"    Summary {scenario['id']}: {hits}/{len(scenario['keywords'])} keywords ({score:.0%})")
    avg = total_score / len(SUMMARY_SCENARIOS)
    return {"score": avg, "details": details}


def bench_infopres() -> dict:
    SUMMARY_PROMPT = (
        "You are a senior engineer summarizing this for an AI orchestrator. "
        "Write a summary under 200 words that preserves ALL specific values, limits, "
        "settings, defaults, yes/no constraints, and technical facts. Use bullet points. "
        "Keep exact numeric values, endpoints, statuses, and configuration names verbatim."
    )
    total_score = 0.0
    details = []
    for scenario in INFO_SCENARIOS:
        try:
            summary = llm([
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": scenario["content"]},
            ], max_tokens=300)
            hits = sum(1 for _, ans in scenario["questions"] if contains_fact(summary, ans))
            score = hits / len(scenario["questions"])
        except Exception:
            score = 0.0
            hits = 0
            summary = ""
        total_score += score
        details.append({
            "scenario": scenario["id"],
            "facts_preserved": hits,
            "facts_total": len(scenario["questions"]),
            "score": round(score, 3),
            "preview": summary[:240],
        })
        print(f"    InfoPres {scenario['id']}: {hits}/{len(scenario['questions'])} facts ({score:.0%})")
    avg = total_score / len(INFO_SCENARIOS)
    return {"score": avg, "details": details}


def measure_tps() -> float:
    """Quick throughput measurement: generate 200 tokens and time it."""
    prompt = (
        "Write a Python class implementing a thread-safe LRU cache with get() and put() "
        "methods. Include complete implementation with all edge cases handled."
    )
    t0 = time.time()
    try:
        raw_payload = json.dumps({
            "model": _api_model_id(),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{LM_STUDIO_URL}/v1/chat/completions",
            data=raw_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        elapsed = time.time() - t0
        tokens = data.get("usage", {}).get("completion_tokens", 200)
        return round(tokens / elapsed, 1)
    except Exception:
        return 0.0


# ─── RESULTS ───────────────────────────────────────────────────────────────────

def composite_score(r: dict) -> float:
    he = r.get("humaneval", {}).get("score", 0)
    mb = r.get("mbpp", {}).get("score", 0)
    bf = r.get("bugfix", {}).get("score", 0)
    sm = r.get("summary", {}).get("score", 0)
    ip = r.get("infopres", {}).get("score", 0)
    return (WEIGHTS["humaneval"] * he + WEIGHTS["mbpp"] * mb + WEIGHTS["bugfix"] * bf +
            WEIGHTS["summary"] * sm + WEIGHTS["infopres"] * ip)


def print_report(all_results: list):
    ranked = sorted(all_results, key=lambda r: r["composite"], reverse=True)
    W = 145
    print("\n" + "═" * W)
    print("  LOCAL CODER / BUG-FIXER BENCHMARK — FINAL RESULTS")
    print("  Role: local coding worker with orchestrator-provided plans")
    print("=" * W)
    hdr = f"  {'#':<3} {'Publisher':<18} {'Model Label':<34} {'Type':<12} {'Score':>6} {'HumanEval':>10} {'MBPP':>6} {'BugFix':>7} {'Summary':>8} {'InfoPres':>9} {'TPS':>6}"
    print(hdr)
    print("  " + "─" * (W - 2))
    for i, r in enumerate(ranked, 1):
        publisher = r.get("publisher") or "unknown"
        print(
            f"  {i:<3} {publisher[:18]:<18} {r['label'][:34]:<34} {r['type'][:12]:<12} "
            f"{r['composite']:>6.1%} {r.get('humaneval',{}).get('score',0):>10.1%} "
            f"{r.get('mbpp',{}).get('score',0):>6.1%} "
            f"{r.get('bugfix',{}).get('score',0):>7.1%} "
            f"{r.get('summary',{}).get('score',0):>8.1%} "
            f"{r.get('infopres',{}).get('score',0):>9.1%} "
            f"{r.get('tps',0):>6.1f}"
        )
    print("═" * W)
    if ranked:
        winner = ranked[0]
        print(f"\n  WINNER: {winner['label']} ({winner['type']}) — composite {winner['composite']:.1%}")
        print("  Best for local coder / bug-fixer role on this LM Studio setup")
    print()


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    global LM_STUDIO_HOST, LM_STUDIO_PORT, LM_STUDIO_URL

    parser = argparse.ArgumentParser(description="Local Coder / Bug-Fixer Benchmark (LM Studio)")
    parser.add_argument("--dry-run", action="store_true", help="List discovered LM Studio models, no benchmark")
    parser.add_argument("--report", action="store_true", help="Print report from latest saved results")
    parser.add_argument("--host", default=LM_STUDIO_HOST, help=f"LM Studio host (default: {LM_STUDIO_HOST})")
    parser.add_argument("--port", type=int, default=LM_STUDIO_PORT, help=f"LM Studio port (default: {LM_STUDIO_PORT})")
    parser.add_argument("--models", default="",
                        help="Comma-separated substrings; only models whose key contains one of them will be tested.")
    args = parser.parse_args()

    # Allow per-run override of the LM Studio endpoint.
    LM_STUDIO_HOST = args.host
    LM_STUDIO_PORT = args.port
    LM_STUDIO_URL = f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Find or create run directory
    progress_files = sorted(RESULTS_DIR.glob(f"{RUN_PREFIX}_*/progress.json"))

    if args.report:
        if not progress_files:
            print("No results found.")
            return
        pfile = progress_files[-1]
        with open(pfile) as f:
            all_results = json.load(f)
        print_report(all_results)
        return

    # ---- Discover models from LM Studio ----
    try:
        ensure_lm_studio_server()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    try:
        models = discover_lm_studio_models()
    except Exception as e:
        print(f"ERROR discovering LM Studio models: {e}")
        sys.exit(1)

    if args.models.strip():
        wanted = [s.strip().lower() for s in args.models.split(",") if s.strip()]
        models = [m for m in models if any(w in m["id"].lower() or w in m["label"].lower() for w in wanted)]

    if not models:
        print("No LM Studio LLMs found. Run `lms ls` to confirm what's downloaded, or open LM Studio "
              "and download at least one model. Use --models to filter; an empty list means no match.")
        sys.exit(1)

    # Re-rank to 1..N after filtering so progress prints stay tidy.
    for i, m in enumerate(models, 1):
        m["rank"] = i

    # Dry run
    if args.dry_run:
        print("DRY RUN — Local Coder / Bug-Fixer Benchmark (LM Studio)")
        print(f"\nLM Studio endpoint: {LM_STUDIO_URL}")
        print(f"lms CLI:            {_lms_bin()}")
        print(f"\n{len(models)} loadable LLM model(s) discovered in LM Studio:")
        for m in models:
            size_gb = (m.get("size_bytes") or 0) / 1e9
            size_str = f"{size_gb:>5.1f} GB" if size_gb else "    ? GB"
            quant = m.get("quantization") or ""
            quant_str = f"{quant:<7}" if quant else "       "
            publisher = m.get("publisher") or "unknown"
            print(f"  {m['rank']:2}. [{m['type']:<10}] {publisher:<18} {quant_str} {size_str}  {m['id']}")
        print(f"\nBenchmark tasks:")
        print(f"  HumanEval:  {len(HUMANEVAL)} problems")
        print(f"  MBPP:       {len(MBPP)} problems")
        print(f"  BugFix:     {len(BUGFIX_SCENARIOS)} QuixBugs-style repair tasks")
        print(f"  Summary:    {len(SUMMARY_SCENARIOS)} scenarios (keywords: {sum(len(s['keywords']) for s in SUMMARY_SCENARIOS)} total)")
        print(f"  InfoPres:   {len(INFO_SCENARIOS)} scenarios (facts: {sum(len(s['questions']) for s in INFO_SCENARIOS)} total)")
        total_prompts = len(HUMANEVAL) + len(MBPP) + len(BUGFIX_SCENARIOS) + len(SUMMARY_SCENARIOS) + len(INFO_SCENARIOS) + 1
        print(f"  Total prompts per model: {total_prompts}")
        print(f"\nEstimated time: {len(models) * 30} min (~{len(models) * 30 / 60:.1f}h) at 30min/model")
        return

    # Set up run directory (resume if exists)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"{RUN_PREFIX}_{ts}"
    # Check for resumable run
    if progress_files:
        latest = progress_files[-1]
        with open(latest) as f:
            previous_results = json.load(f)
        current_model_ids = {m["id"] for m in models}
        previous_results = [
            r for r in previous_results
            if r.get("model_key") in current_model_ids
        ]
        completed = {
            r["model_key"] for r in previous_results
            if r.get("error") is None
            and r.get("model_key")
            and r.get("publisher")
        }
        remaining = [m for m in models if m["id"] not in completed]
        if remaining and len(remaining) < len(models):
            print(f"Resuming from {latest.parent.name} — {len(completed)} done, {len(remaining)} remaining")
            run_dir = latest.parent
            all_results = previous_results
        else:
            run_dir.mkdir(parents=True)
            all_results = []
            completed = set()
    else:
        run_dir.mkdir(parents=True)
        all_results = []
        completed = set()

    progress_file = run_dir / "progress.json"

    print(f"Local Coder / Bug-Fixer Benchmark — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Backend: LM Studio @ {LM_STUDIO_URL}")
    print(f"Results: {run_dir}")
    print(f"Models: {len(models)} | Completed: {len(completed)}\n")

    for model in models:
        if model["id"] in completed:
            print(f"[{model['rank']:2}/{len(models)}] SKIP (done): {model['label']}")
            continue

        print(f"\n{'─'*70}")
        print(f"[{model['rank']:2}/{len(models)}] {model['label']}  [{model['type']}]")
        print(f"       {model['id']}")
        print("─" * 70)

        loaded = False
        result = {
            "model_key": model["id"],
            "base_key": model.get("base_key", model["id"]),
            "label": model["label"],
            "publisher": model.get("publisher", ""),
            "path": model.get("path", ""),
            "type": model["type"],
            "format": model.get("format", ""),
            "quantization": model.get("quantization", ""),
            "selected_variant": model.get("selected_variant", ""),
            "available_variants": model.get("available_variants", []),
            "params": model.get("params", ""),
            "max_context": model.get("max_context", 0),
            "size_gb": round((model.get("size_bytes") or 0) / 1e9, 2),
            "rank": model["rank"],
        }
        log_path = run_dir / f"lms_{model['rank']:02d}.log"

        try:
            # 1. Load model into LM Studio
            t_start = time.time()
            load_model(model["id"], log_path=log_path)
            loaded = True
            result["api_model_id"] = _api_model_id()
            print(f"  Model ready in {time.time() - t_start:.1f}s")

            # 2. Measure TPS
            print("  Measuring TPS...")
            result["tps"] = measure_tps()
            print(f"  TPS: {result['tps']:.1f} tok/s")

            # 3. HumanEval
            print("  Running HumanEval (20 problems)...")
            result["humaneval"] = bench_humaneval()
            print(f"  HumanEval: {result['humaneval']['passed']}/20 = {result['humaneval']['score']:.0%}")

            # 4. MBPP
            print("  Running MBPP (20 problems)...")
            result["mbpp"] = bench_mbpp()
            print(f"  MBPP: {result['mbpp']['passed']}/20 = {result['mbpp']['score']:.0%}")

            # 5. Bug fixing
            print(f"  Running BugFix ({len(BUGFIX_SCENARIOS)} QuixBugs-style tasks)...")
            result["bugfix"] = bench_bugfix()
            print(f"  BugFix: {result['bugfix']['passed']}/{len(BUGFIX_SCENARIOS)} = {result['bugfix']['score']:.0%}")

            # 6. Summary Accuracy
            print("  Running Summary Accuracy (5 scenarios)...")
            result["summary"] = bench_summary()
            print(f"  Summary: {result['summary']['score']:.0%}")

            # 7. Info Preservation
            print("  Running Info Preservation (5 scenarios)...")
            result["infopres"] = bench_infopres()
            print(f"  InfoPres: {result['infopres']['score']:.0%}")

            result["error"] = None

        except Exception as e:
            print(f"  ERROR: {e}")
            result["error"] = str(e)
            result.setdefault("humaneval", {"score": 0})
            result.setdefault("mbpp", {"score": 0})
            result.setdefault("bugfix", {"score": 0})
            result.setdefault("summary", {"score": 0})
            result.setdefault("infopres", {"score": 0})
            result.setdefault("tps", 0.0)

        finally:
            if loaded:
                try:
                    unload_model(model["id"], log_path=log_path)
                except Exception as e:
                    print(f"  WARN: unload failed: {e}")

        result["composite"] = composite_score(result)
        print(f"  Composite Score: {result['composite']:.1%}")
        all_results = [r for r in all_results if r.get("model_key") != model["id"]]
        all_results.append(result)
        completed.add(model["id"])

        # Save progress after every model
        with open(progress_file, "w") as f:
            json.dump(all_results, f, indent=2)

    print_report(all_results)

    # Save final report text
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_report(all_results)
    (run_dir / "final_report.txt").write_text(buf.getvalue())
    print(f"Report saved: {run_dir / 'final_report.txt'}")


if __name__ == "__main__":
    main()
