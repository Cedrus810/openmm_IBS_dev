from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Tuple

import pytest


ROOT = Path(__file__).resolve().parents[1]
PREOPT_PATH = ROOT / "abfe_preoptimizer.py"


def _load_partition_function():
    tree = ast.parse(PREOPT_PATH.read_text(encoding="utf-8"), filename=str(PREOPT_PATH))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_greedy_vanishing_window_ranges"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"List": List, "Tuple": Tuple}
    exec(compile(module, str(PREOPT_PATH), "exec"), namespace)
    return namespace["_greedy_vanishing_window_ranges"]


def _assert_partition_invariants(ranges, n_states, min_states, max_states):
    assert ranges[0][0] == 0
    assert ranges[-1][1] == n_states
    assert all(min_states <= end - start <= max_states for start, end in ranges)
    assert all(left[1] - 1 == right[0] for left, right in zip(ranges, ranges[1:]))
    assert all(
        right[0] >= left[1]
        for i, left in enumerate(ranges)
        for right in ranges[i + 2 :]
    )
    assert {state for start, end in ranges for state in range(start, end)} == set(
        range(n_states)
    )


def test_infeasible_exact_six_state_windows_are_rejected():
    partition = _load_partition_function()
    with pytest.raises(ValueError, match="不存在满足"):
        partition(7, 6, 6)


@pytest.mark.parametrize(
    "n_states,min_states,max_states",
    [(2, 2, 2), (12, 4, 6), (23, 4, 6), (31, 3, 7)],
)
def test_feasible_partitions_satisfy_every_window_invariant(
    n_states, min_states, max_states
):
    partition = _load_partition_function()
    ranges = partition(n_states, min_states, max_states)
    _assert_partition_invariants(ranges, n_states, min_states, max_states)
    sizes = [end - start for start, end in ranges]
    assert sizes == sorted(sizes)

