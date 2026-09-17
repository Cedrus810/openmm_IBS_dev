"""Stage-2 legacy rescue must not take control after the autonomous loop."""

import ast
from pathlib import Path
import pytest

pytestmark = pytest.mark.cpu_only


PIPELINE = Path(__file__).resolve().parents[1] / "abfe_pipeline.py"


def test_bridge_rescue_ownership_truth_table():
    tree = ast.parse(PIPELINE.read_text(encoding="utf-8"))
    assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_bridge_rescue_enabled"
                for target in node.targets)
    ]
    assert len(assignments) == 1
    expression = compile(ast.Expression(assignments[0].value), str(PIPELINE), "eval")
    for autonomous in (False, True):
        for configured in (False, True):
            actual = eval(expression, {"kwargs": {"stage2_enable_bridge_rescue": configured},
                                       "_autonomous_on": autonomous})
            assert actual is (configured and not autonomous)

    rescue_guards = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(isinstance(n, ast.Name) and n.id == "_bridge_rescue_enabled"
                for n in ast.walk(node.test))
        and any(isinstance(n, ast.Constant) and n.value == "vanishing_rescue"
                for n in ast.walk(node))
    ]
    assert len(rescue_guards) == 1
