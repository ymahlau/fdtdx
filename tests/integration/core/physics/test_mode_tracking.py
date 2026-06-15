"""Integration tests for field-overlap mode tracking in compute_mode.

Geometry: three-layer waveguide (eps=4 core, eps=8 top/bottom slabs) in a 6 um x 6 um
cross-section, propagation along z.  At mode_index=9 and frequencies spanning
lambda ~1.5-2.5 um, neff branches cross: independent per-frequency neff-rank sorting
returns a different physical mode at the crossing step (field overlap ~0.63).

``reference_E`` (scalar frequency) and automatic tracking (list frequency) both
maintain physical continuity across the crossing (consecutive overlap > 0.70).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.core.physics.modes import compute_mode

_C0 = 299_792_458.0

_LAMBDA0_UM = 2.0
_FREQ0 = _C0 / (_LAMBDA0_UM * 1e-6)
_FWIDTH = _FREQ0 / 3.0
_FREQS = np.linspace(_FREQ0 - _FWIDTH, _FREQ0 + _FWIDTH, 11)

# Steps 8→9 straddle a mode crossing for mode_index=9 (lambda ≈ 1.667 → 1.579 µm).
_FREQ_BEFORE_CROSSING = _FREQS[8]
_FREQ_AT_CROSSING = _FREQS[9]
_CROSSING_MODE = 9

_RES = 0.15e-6
_NCELLS = 40


def _make_inv_eps() -> jnp.ndarray:
    """Inverse-permittivity for a three-layer waveguide, shape (3, 40, 40, 1).

    Propagation along z; cross-section in x-y.

        Main core:    eps=4, 1.5 µm wide (10 cells), 1.0 µm tall (7 cells)
        Bottom layer: eps=8, same width,  1.0 µm tall, centre at y = -0.8 µm
        Top layer:    eps=8, same width,  0.8 µm tall, centre at y = +0.7 µm
    """
    mid = _NCELLS // 2
    eps = np.ones((_NCELLS, _NCELLS), dtype=np.float32)

    wx = round(1.5e-6 / _RES)
    wy = round(1.0e-6 / _RES)
    eps[mid - wx // 2 : mid + wx // 2, mid - wy // 2 : mid + wy // 2] = 4.0

    by_c = mid - round(0.8e-6 / _RES)
    by_h = round(1.0e-6 / _RES)
    eps[mid - wx // 2 : mid + wx // 2, by_c - by_h // 2 : by_c + by_h // 2] = 8.0

    ty_c = mid + round(0.7e-6 / _RES)
    ty_h = round(0.8e-6 / _RES)
    eps[mid - wx // 2 : mid + wx // 2, ty_c - ty_h // 2 : ty_c + ty_h // 2] = 8.0

    inv_eps_2d = 1.0 / eps
    inv_eps = np.broadcast_to(inv_eps_2d[np.newaxis, :, :, np.newaxis], (3, _NCELLS, _NCELLS, 1)).copy()
    return jnp.array(inv_eps)


def _field_overlap(E1: np.ndarray, E2: np.ndarray) -> float:
    """Normalised |<E1|E2>| overlap in [0, 1]."""
    denom = np.sqrt(np.sum(np.abs(E1) ** 2) * np.sum(np.abs(E2) ** 2)) + 1e-30
    return float(np.abs(np.sum(np.conj(E1) * E2)) / denom)


@pytest.fixture(scope="module")
def inv_eps():
    return _make_inv_eps()


@pytest.fixture(scope="module")
def mode_before_crossing(inv_eps):
    E, H, neff = compute_mode(
        _FREQ_BEFORE_CROSSING,
        inv_eps,
        1.0,
        resolution=_RES,
        mode_index=_CROSSING_MODE,
    )
    return np.array(E), np.array(H), neff


class TestFieldOverlapModeTracking:
    """``reference_E`` in ``compute_mode`` prevents mode hopping at a branch crossing.

    At the crossing step, neff-rank sorting alone returns a different physical mode
    (field overlap ≈ 0.63).  Passing the previous step's mode_E as ``reference_E``
    selects the candidate with the highest field overlap, maintaining continuity.
    """

    def test_without_reference_E_overlap_is_low_at_crossing(self, inv_eps, mode_before_crossing):
        """neff-rank selection alone gives field overlap < 0.70 at the crossing step."""
        E_before, _, _ = mode_before_crossing
        E_after, _, _ = compute_mode(
            _FREQ_AT_CROSSING,
            inv_eps,
            1.0,
            resolution=_RES,
            mode_index=_CROSSING_MODE,
        )
        overlap = _field_overlap(E_before, np.array(E_after))
        assert overlap < 0.70, (
            f"Expected overlap < 0.70 at crossing without tracking, "
            f"got {overlap:.3f} — geometry may not produce a crossing at these frequencies"
        )

    def test_with_reference_E_overlap_is_higher_at_crossing(self, inv_eps, mode_before_crossing):
        """``reference_E`` raises field overlap above 0.70 at the crossing step."""
        E_before, _, _ = mode_before_crossing
        E_tracked, _, _ = compute_mode(
            _FREQ_AT_CROSSING,
            inv_eps,
            1.0,
            resolution=_RES,
            mode_index=_CROSSING_MODE,
            reference_E=E_before,
        )
        overlap = _field_overlap(E_before, np.array(E_tracked))
        assert overlap > 0.70, f"Expected reference_E tracking to raise overlap above 0.70, got {overlap:.3f}"


class TestComputeModeMultiFreqFullSweep:
    """``compute_mode`` with list frequency maintains field continuity across all 11 frequencies.

    Each consecutive pair of mode fields must have overlap > 0.70, including
    the crossing at steps 8→9 where neff-rank sorting alone drops to ~0.63.
    """

    def test_all_consecutive_overlaps_above_threshold(self, inv_eps):
        inv_eps_stack = jnp.stack([inv_eps] * len(_FREQS), axis=0)
        mode_Es, _, _ = compute_mode(
            frequency=list(_FREQS),
            inv_permittivities=inv_eps_stack,
            inv_permeabilities=1.0,
            resolution=_RES,
            mode_index=_CROSSING_MODE,
        )
        mode_Es_np = np.array(mode_Es)
        for i in range(len(_FREQS) - 1):
            overlap = _field_overlap(mode_Es_np[i], mode_Es_np[i + 1])
            assert overlap > 0.70, f"Steps {i}→{i + 1}: overlap {overlap:.3f} < 0.70"
