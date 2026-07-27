import json
import os
import tempfile
import unittest
from pathlib import Path

import runabfe


ROOT = Path(__file__).resolve().parent


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
        branch = self.runner.index("if config.only_complex_charging:")
        full_run = self.runner.index(
            "complex_results = pipeline.run_full_pipeline(", branch
        )
        between = self.runner[branch:full_run]
        self.assertIn("_run_complex_charging_only(", between)
        self.assertIn("return", between)

    def test_charging_only_bypasses_generic_boresch_resolver(self):
        frozen_branch = self.runner.index(
            "boresch_restraint = _load_frozen_stage2_boresch(output_dir)"
        )
        generic_resolver = self.runner.index(
            "boresch_restraint = resolve_boresch_restraint(config, pipeline)",
            frozen_branch,
        )
        between = self.runner[frozen_branch:generic_resolver]
        self.assertIn("else:", between)

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
