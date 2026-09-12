import ast
import json
import os
import tempfile
import unittest
from pathlib import Path

import runabfe


import pytest

pytestmark = pytest.mark.cpu_only

ROOT = Path(__file__).resolve().parents[1]


class ChargingOnlyPureHelperTests(unittest.TestCase):
    def test_frozen_stage_loader_accepts_finite_matching_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "stage2.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "stage": "vanishing",
                        "total_delta_G": 145.9,
                        "total_error": 1.4,
                    },
                    handle,
                )
            result = runabfe._load_frozen_stage_result(path, "vanishing")
            self.assertEqual(result["stage"], "vanishing")

    def test_frozen_stage_loader_rejects_wrong_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "stage2.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "stage": "decharging",
                        "total_delta_G": 1.0,
                        "total_error": 0.1,
                    },
                    handle,
                )
            with self.assertRaises(RuntimeError):
                runabfe._load_frozen_stage_result(path, "vanishing")

    def test_rerun_dir_rejects_nonempty_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "candidate")
            os.makedirs(target)
            with open(os.path.join(target, "keep.txt"), "w", encoding="utf-8") as handle:
                handle.write("do not overwrite")
            with self.assertRaises(FileExistsError):
                runabfe._prepare_charging_rerun_dir(tmp, target)

    def test_boresch_signature_ignores_non_hamiltonian_diagnostics(self):
        base = {
            "receptor_indices": [1, 2, 3],
            "ligand_indices": [4, 5, 6],
            "equilibrium_values": {"r0": 0.5},
            "force_constants": {"kr": 2000.0},
        }
        other = dict(base, diagnostics={"note": "different provenance only"})
        self.assertEqual(
            runabfe._boresch_core_signature(base),
            runabfe._boresch_core_signature(other),
        )

    def test_frozen_stage2_boresch_is_loaded_from_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = os.path.join(tmp, "checkpoints")
            os.makedirs(checkpoint_dir)
            path = os.path.join(checkpoint_dir, "stage2_vanishing.json")
            params = {
                "receptor_indices": [1, 2, 3],
                "ligand_indices": [4, 5, 6],
                "equilibrium_values": {
                    "r0": 0.47,
                    "thetaA0": 1.2,
                    "thetaB0": 1.3,
                    "phiA0": 0.1,
                    "phiB0": 0.2,
                    "phiC0": 0.3,
                },
                "force_constants": {
                    "kr": 2000.0,
                    "kthetaA": 200.0,
                    "kthetaB": 200.0,
                    "kphiA": 200.0,
                    "kphiB": 150.0,
                    "kphiC": 120.0,
                },
            }
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "stage": "vanishing",
                        "total_delta_G": 145.9,
                        "total_error": 1.4,
                        "protocol_key": {"payload": {"boresch_params": params}},
                    },
                    handle,
                )
            loaded = runabfe._load_frozen_stage2_boresch(tmp)
            self.assertEqual(loaded["equilibrium_values"]["r0"], 0.47)


class ChargingOnlySourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = (ROOT / "runabfe.py").read_text(encoding="utf-8")
        cls.pipeline = (ROOT / "abfe_pipeline.py").read_text(encoding="utf-8")

    def test_main_returns_before_full_pipeline_in_charging_only_mode(self):
        """`--only-complex-charging` 分支必须调完隔离入口就 return。

        2026-09-09 重写：原实现在两个锚点之间的**源码文本**里搜 `"return"`。
        `return` 这个词在 runabfe.py 的任意一段里几乎必然出现（任何嵌套函数、
        任何注释），所以那条断言实际上不可能失败 —— 它没有测到"提前返回"。
        现在用 AST：找到 `if config.only_complex_charging:` 那个 If，要求它的
        分支体里既调用了隔离入口、又**以 Return 结束**。
        """
        tree = ast.parse(self.runner, filename="runabfe.py")
        branches = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and "only_complex_charging" in ast.unparse(node.test)
        ]
        self.assertTrue(branches, "找不到 only_complex_charging 分支")

        matched = []
        for node in branches:
            called = {
                getattr(c.func, "attr", None) or getattr(c.func, "id", None)
                for c in ast.walk(node) if isinstance(c, ast.Call)
            }
            if "_run_complex_charging_only" not in called:
                continue
            matched.append(node)
            # 分支体的最后一条可执行语句必须是 return —— 这才是"不落进完整链路"。
            self.assertIsInstance(
                node.body[-1], ast.Return,
                "only_complex_charging 分支没有以 return 结束，会继续落进 "
                "run_full_pipeline（完整 dual_lambda 链路）",
            )
            # 且该分支内部不得直接触发完整链路。
            self.assertNotIn(
                "run_full_pipeline", called,
                "隔离模式的分支里直接调了 run_full_pipeline",
            )
        self.assertTrue(
            matched, "没有任何 only_complex_charging 分支调用 _run_complex_charging_only"
        )

    def test_charging_only_bypasses_generic_boresch_resolver(self):
        """frozen 分支与通用 resolver 必须是**同一个 if/else 的两个互斥支**。

        2026-09-09 重写：原实现在两个锚点之间搜 `"else:"`。那段区间里任何一个
        无关的 else（另一个 if、一个 try/except/else）都能满足它，所以"两者互斥"
        这个真正的契约没有被测到。现在用 AST 直接确认它们分居同一个 If 的
        body / orelse。
        """
        tree = ast.parse(self.runner, filename="runabfe.py")

        def _calls(node):
            return {
                getattr(c.func, "attr", None) or getattr(c.func, "id", None)
                for c in ast.walk(node) if isinstance(c, ast.Call)
            }

        exclusive = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.If) or not node.orelse:
                continue
            body_calls = set().union(*(_calls(n) for n in node.body)) if node.body else set()
            else_calls = set().union(*(_calls(n) for n in node.orelse)) if node.orelse else set()
            frozen_in_body = "_load_frozen_stage2_boresch" in body_calls
            generic_in_else = "resolve_boresch_restraint" in else_calls
            frozen_in_else = "_load_frozen_stage2_boresch" in else_calls
            generic_in_body = "resolve_boresch_restraint" in body_calls
            if (frozen_in_body and generic_in_else) or (frozen_in_else and generic_in_body):
                exclusive.append(node.lineno)

        self.assertTrue(exclusive, (
            "找不到把 _load_frozen_stage2_boresch 与 resolve_boresch_restraint 放在"
            "同一个 if/else 两支上的结构 —— 两者不再互斥，冻结 stage2 的 Boresch "
            "可能被通用 resolver 覆盖"
        ))

    def test_isolated_stage_forces_fresh_sampling(self):
        start = self.runner.index("def _run_complex_charging_only(")
        end = self.runner.index("\n\n# 主入口", start)
        body = self.runner[start:end]
        self.assertIn('"decharging"', body)
        self.assertIn("resume=False", body)
        self.assertNotIn("run_full_pipeline(", body)
        self.assertNotIn("optimize_stage2", body)

    def test_max_resident_contexts_is_forwarded_to_remd(self):
        self.assertIn(
            "remd_max_resident_contexts: Optional[int] = None",
            self.pipeline,
        )
        self.assertIn(
            "max_resident_contexts=remd_max_resident_contexts",
            self.pipeline,
        )


if __name__ == "__main__":
    unittest.main()
