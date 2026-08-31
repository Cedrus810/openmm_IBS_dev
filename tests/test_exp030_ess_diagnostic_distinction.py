"""Deterministic (no GPU) regression test proving `ibs_engine.py`'s two ESS
diagnostics are genuinely different quantities that can diverge sharply --
not interchangeable, and not safe to cite as "the real target-reweighting
overlap" without checking which one you have.

Background (2026-08-27 review): earlier analysis in this investigation cited
`window_overlap_diagnostics["min_ess_ratio"]` as "the real overlap of the
sampled distribution reweighted onto the physical target state" -- that was
wrong. Reading `_ibs_reweighting_quality_diagnostics` (ibs_engine.py) shows
`mixture_ess`/`min_ess_ratio` is computed ENTIRELY from `u_kj_raw` and the
frozen `f_k` via a theoretical softmax reconstruction:

    logits = -(u_kj_raw - f_k) / kt
    p_k(x_n) = softmax_k(logits)[k, n]

It never reads the actually-MEASURED `bias_kj` at all -- it reports what
overlap WOULD look like if the true sampled bias exactly matched this
f_k-based reconstruction, not what the overlap actually was. The literal,
measured-data quantity is `raw_ess`/`raw_ess_ratio` (also computed in the
same function, and independently again elsewhere via
`mbar.compute_effective_sample_number()` on the augmented matrix as
`raw_min_ess_ratio` -- see docs/experiments/EXP-030_FINAL_STATUS_2026-08-27.md section 5g),
which DOES use the measured `bias_kj`:

    log_w_raw = (bias_kj - u_kj_raw) / kt

This test constructs a synthetic case where the real measured bias includes
an extra per-frame, state-independent term the f_k-based reconstruction
never sees (exactly the shape of a real Group-4 lambda-WCA guard-shell
spike found in window_2 of the real production data) -- and shows
`mixture_ess_ratio` stays blind to it (falsely reports good overlap) while
`raw_ess_ratio` correctly collapses.
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ibs_engine import _ibs_reweighting_quality_diagnostics  # noqa: E402


class EssDiagnosticDistinctionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20260827)
        self.kt = 0.00831446261815324 * 300.0
        self.n_states = 3
        self.n_frames = 1000
        # Mild state-dependent spread with small per-frame noise -- a
        # perfectly benign, well-overlapped system on its own.
        state_means = np.array([0.0, 5.0, 10.0])
        self.u_kj_raw = state_means[:, None] + rng.normal(0.0, 1.0, size=(self.n_states, self.n_frames))
        self.f_k = np.array([0.0, -5.0, -10.0])  # roughly flattens the mixture by construction

        # The bias a perfectly-matching mixture (bias == theoretical
        # reconstruction, nothing extra) would have produced.
        logits = -(self.u_kj_raw - self.f_k[:, None]) / self.kt
        log_norm = np.logaddexp.reduce(logits, axis=0)
        self.theoretical_bias = -self.kt * log_norm

        # A real Group-4-like guard spike: state-independent (same value
        # added to every state at that frame), rare, large, at known frames
        # -- exactly the shape found in the real repeat2/window_2 audit.
        self.guard_spike_frames = [7, 402, 861]
        self.guard_spike_kt = 12.0
        guard = np.zeros(self.n_frames)
        guard[self.guard_spike_frames] = self.guard_spike_kt * self.kt
        self.measured_bias = self.theoretical_bias + guard

    def test_mixture_ess_is_blind_to_a_real_guard_spike_the_raw_one_catches(self):
        clean = _ibs_reweighting_quality_diagnostics(self.u_kj_raw, self.theoretical_bias, self.f_k, self.kt)
        spiked = _ibs_reweighting_quality_diagnostics(self.u_kj_raw, self.measured_bias, self.f_k, self.kt)

        # mixture_ess never reads bias_kj at all -- adding the guard spike to
        # the measured bias must not move it even slightly.
        np.testing.assert_array_equal(clean["mixture_ess_ratio"], spiked["mixture_ess_ratio"])

        # raw_ess DOES read bias_kj -- the guard spike must visibly degrade it.
        for k in range(self.n_states):
            self.assertLess(
                spiked["raw_ess_ratio"][k], clean["raw_ess_ratio"][k] * 0.5,
                f"state {k}: raw_ess_ratio should collapse once the measured bias carries "
                "the guard spike -- if this fails, raw_ess_ratio stopped reading bias_kj",
            )

        # The two diagnostics must now disagree sharply on the spiked data --
        # this is the literal "min_ess_ratio looked fine, raw_min_ess_ratio
        # did not" failure mode from the real window_2 audit, reproduced
        # from first principles.
        for k in range(self.n_states):
            self.assertGreater(
                spiked["mixture_ess_ratio"][k] - spiked["raw_ess_ratio"][k], 0.3,
                f"state {k}: mixture_ess_ratio should be reporting a much rosier picture "
                "than raw_ess_ratio once a real, bias-only guard spike is present",
            )

    def test_top1pct_raw_weight_flags_the_same_spike_frames(self):
        spiked = _ibs_reweighting_quality_diagnostics(self.u_kj_raw, self.measured_bias, self.f_k, self.kt)
        clean = _ibs_reweighting_quality_diagnostics(self.u_kj_raw, self.theoretical_bias, self.f_k, self.kt)
        # top1pct_raw_weight is also derived from the raw (bias-using) path --
        # it must rise sharply once 3/1000 frames each carry a 12 kT spike.
        for k in range(self.n_states):
            self.assertGreater(spiked["top1pct_raw_weight"][k], clean["top1pct_raw_weight"][k] + 0.2)


if __name__ == "__main__":
    unittest.main()
