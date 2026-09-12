"""Validate proposed audit diffs without changing the working tree.

Usage: python tools/diagnostics/check_audit_patches.py patch1.patch [...]
Checks hunk contents, compiles each patched Python source in memory, and runs
the dependency-light refinement regression against that in-memory source.
"""
import argparse
import ast
from pathlib import Path
import re
import runpy
import unittest


def patched_sources(root, patches):
    sources = {}
    for patch in patches:
        lines = patch.read_text(encoding="utf-8-sig").splitlines(keepends=True)
        i = 0
        path = None
        offset = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("+++ b/"):
                path = line[6:].strip()
                if path not in sources:
                    target = root / path
                    sources[path] = (target.read_text(encoding="utf-8-sig")
                                     if target.exists() else "")
                offset = 0
                i += 1
                continue
            match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if match:
                if path is None:
                    raise ValueError("Hunk before file header")
                old_start = int(match.group(1))
                old_count = int(match.group(2) or 1)
                new_count = int(match.group(4) or 1)
                old, new = [], []
                i += 1
                while i < len(lines) and lines[i][:1] in (" ", "+", "-"):
                    if lines[i].startswith(("--- a/", "+++ b/")):
                        break
                    prefix, value = lines[i][0], lines[i][1:]
                    if prefix in (" ", "-"):
                        old.append(value)
                    if prefix in (" ", "+"):
                        new.append(value)
                    i += 1
                if len(old) != old_count or len(new) != new_count:
                    raise ValueError(f"{path}: invalid hunk counts at {old_start}")
                content = sources[path].splitlines(keepends=True)
                start = max(0, old_start - 1) + offset
                if content[start:start + old_count] != old:
                    # Earlier independent diffs may have shifted the position.
                    matches = [j for j in range(len(content) - old_count + 1)
                               if content[j:j + old_count] == old]
                    if len(matches) != 1:
                        raise ValueError(f"{path}: hunk does not match uniquely")
                    start = matches[0]
                content[start:start + old_count] = new
                sources[path] = "".join(content)
                offset += new_count - old_count
                continue
            i += 1
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patches", nargs="+", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sources = patched_sources(root, args.patches)
    for path, source in sources.items():
        if path.endswith(".py"):
            ast.parse(source, filename=path)
            print(f"Syntax OK: {path}")
    if "abfe_preoptimizer.py" in sources:
        namespace = runpy.run_path(str(root / "tests/test_stage2_refinement_restart.py"))
        case = namespace["RefinementRestartTests"]
        case.source = sources["abfe_preoptimizer.py"]
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(case)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)


if __name__ == "__main__":
    main()
