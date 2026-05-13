"""
Run the test suite with a small stdlib-only coverage gate.

This avoids requiring pytest/coverage in the user's local Python install.
The coverage target is the betting engine, which is the core logic most likely
to affect predictions, hit rate, and ROI.
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
import trace
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
TESTS = os.path.join(ROOT, "tests")
TARGETS = [os.path.join(SRC, "betting.py")]


def executable_lines(path: str) -> set[int]:
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source, filename=path)
    lines: set[int] = set()
    function_ranges = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_ranges.append((node.lineno, node.end_lineno or node.lineno))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.stmt, ast.ExceptHandler)):
            continue
        start = getattr(node, "lineno", None)
        if start is None:
            continue
        if not any(fn_start < start <= fn_end for fn_start, fn_end in function_ranges):
            continue
        lines.add(start)
    return lines


def run_suite() -> unittest.result.TestResult:
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    suite = unittest.defaultTestLoader.discover(TESTS)
    return unittest.TextTestRunner(verbosity=2).run(suite)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-under", type=float, default=80.0)
    args = parser.parse_args()

    tracer = trace.Trace(count=True, trace=False, ignoredirs=[sys.base_prefix, sys.exec_prefix])
    result = tracer.runfunc(run_suite)
    if not result.wasSuccessful():
        return 1

    counts = tracer.results().counts
    total_lines = 0
    covered_lines = 0
    print("\nCoverage")
    print("--------")
    for target in TARGETS:
        executable = executable_lines(target)
        executed = {
            line_no
            for (filename, line_no), count in counts.items()
            if os.path.abspath(filename) == os.path.abspath(target) and count > 0
        }
        covered = executable & executed
        percent = (len(covered) / len(executable) * 100.0) if executable else 100.0
        total_lines += len(executable)
        covered_lines += len(covered)
        rel = os.path.relpath(target, ROOT)
        print(f"{rel}: {percent:.1f}% ({len(covered)}/{len(executable)})")

    total_percent = (covered_lines / total_lines * 100.0) if total_lines else 100.0
    print(f"TOTAL: {total_percent:.1f}% ({covered_lines}/{total_lines})")
    if total_percent < args.fail_under:
        print(f"ERROR: coverage {total_percent:.1f}% is below {args.fail_under:.1f}%")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
