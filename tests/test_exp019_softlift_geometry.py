"""R1 geometry support-domain tests."""

import pytest

np = pytest.importorskip("numpy")

from local_residual.softlift_dataset import SoftLiftDatasetError, _unit_shift_and_displacement  # noqa: E402


def test_triclinic_unit_shift_and_half_box_tie_fail_closed():
    box = np.array([[10.0, 0.0, 0.0], [2.0, 9.0, 0.0], [1.0, 1.0, 11.0]])
    displacement, shift = _unit_shift_and_displacement(np.zeros(3), np.array([9.0, 0.0, 0.0]), box)
    assert displacement.shape == (3,)
    assert shift.dtype == np.int64
    with pytest.raises(SoftLiftDatasetError):
        _unit_shift_and_displacement(np.zeros(3), 0.5 * box[0], box)
