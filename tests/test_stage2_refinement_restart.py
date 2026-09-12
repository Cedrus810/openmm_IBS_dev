"""Dependency-light behavior tests for Stage 2 refinement restarts."""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional
import unittest

import numpy as np


def load_refiner(source=None):
    if source is None:
        source = (Path(__file__).resolve().parents[1] /
                  "abfe_preoptimizer.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    method = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "_refine_pilot_grid_in_steep_segments")
    namespace = dict(np=np, Optional=Optional, List=List,
                     guarded_step=lambda *args: None)
    exec(compile(ast.Module(body=[method], type_ignores=[]),
                 "<refinement>", "exec"), namespace)
    return namespace


class Context:
    def __init__(self):
        self.params = {"vdw": 0.0, "coul": 0.0}
        self.state = "collapsed"
        self.restored = []

    def getParameters(self):
        return self.params

    def setParameter(self, key, value):
        self.params[key] = value

    def getIntegrator(self):
        return object()

    def setState(self, state):
        self.state = state
        self.restored.append(state)

    def getState(self, **kwargs):
        assert kwargs == dict(getPositions=True, getVelocities=True,
                              getParameters=True)
        return self.state


class RefinementRestartTests(unittest.TestCase):
    source = None

    def test_later_round_restores_inserted_endpoint(self):
        ns = load_refiner(self.source)
        segment_lengths = iter([np.array([9., 1.]), np.array([1., 9., 1.])])
        ns["_pilot_segment_lengths"] = lambda *args: next(segment_lengths)
        context = Context()
        starts = []

        def sample(parameter, lam, **kwargs):
            starts.append((lam, context.state))
            context.state = f"sampled-{lam}"
            return 4., {}

        pilot = SimpleNamespace(context=context, param_vdw="vdw",
                                param_coul="coul", _sample_scalar_metric=sample)
        lam, metric, points = ns["_refine_pilot_grid_in_steep_segments"](
            pilot, [1., .5, 0.], [1., 1., 1.], [{}, {}, {}],
            100, .01, extra_points_per_segment=1, max_rounds=2,
            pilot_states=["sampled-1.0", "sampled-0.5", "sampled-0.0"],
        )
        self.assertEqual(context.restored, ["sampled-1.0", "sampled-0.75"])
        self.assertEqual(starts, [(.75, "sampled-1.0"), (.625, "sampled-0.75")])
        np.testing.assert_array_equal(lam, [1., .75, .625, .5, 0.])
        self.assertEqual(len(metric), len(points))
        self.assertEqual(context.params["coul"], 0.)

    def test_missing_snapshot_fails_before_lambda_jump(self):
        ns = load_refiner(self.source)
        ns["_pilot_segment_lengths"] = lambda *args: np.array([9., 1.])
        context = Context()
        pilot = SimpleNamespace(context=context, param_vdw="vdw", param_coul="coul")
        with self.assertRaisesRegex(ValueError, "endpoint states"):
            ns["_refine_pilot_grid_in_steep_segments"](
                pilot, [1., .5, 0.], [1., 1., 1.], [{}, {}, {}], 100, .01,
            )
        self.assertEqual(context.params["vdw"], 0.)
        self.assertEqual(context.restored, [])


if __name__ == "__main__":
    unittest.main()
