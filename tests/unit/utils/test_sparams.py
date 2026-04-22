"""Unit tests for fdtdx.utils.sparams - pure-logic, no JAX."""

from dataclasses import is_dataclass

import pytest

from fdtdx.utils.sparams import PortSpec, _make_port_shape

# ---------------------------------------------------------------------------
# PortSpec
# ---------------------------------------------------------------------------


class TestPortSpec:
    def test_defaults(self):
        p = PortSpec(center=(0.0, 0.0, 0.0), axis=0, direction="+", width=1.0, height=1.0)
        assert p.mode_index == 0
        assert p.filter_pol == "te"
        assert p.name == ""

    def test_all_fields_stored(self):
        p = PortSpec(
            center=(1.0, 2.0, 3.0),
            axis=2,
            direction="-",
            width=0.5,
            height=0.8,
            mode_index=3,
            filter_pol="tm",
            name="myport",
        )
        assert p.center == (1.0, 2.0, 3.0)
        assert p.axis == 2
        assert p.direction == "-"
        assert p.width == 0.5
        assert p.height == 0.8
        assert p.mode_index == 3
        assert p.filter_pol == "tm"
        assert p.name == "myport"

    def test_filter_pol_none_accepted(self):
        p = PortSpec(center=(0.0, 0.0, 0.0), axis=1, direction="+", width=1.0, height=1.0, filter_pol=None)
        assert p.filter_pol is None

    def test_is_dataclass(self):
        assert is_dataclass(PortSpec)


# ---------------------------------------------------------------------------
# _make_port_shape
# ---------------------------------------------------------------------------


class TestMakePortShape:
    """Verify thickness-axis placement for each propagation axis.

    axis=0 → (resolution, width, height)
    axis=1 → (width, resolution, height)
    axis=2 → (width, height, resolution)
    """

    def test_axis0(self):
        result = _make_port_shape(axis=0, resolution=50e-9, width=400e-9, height=600e-9)
        assert result == (50e-9, 400e-9, 600e-9)

    def test_axis1(self):
        result = _make_port_shape(axis=1, resolution=50e-9, width=400e-9, height=600e-9)
        assert result == (400e-9, 50e-9, 600e-9)

    def test_axis2(self):
        result = _make_port_shape(axis=2, resolution=50e-9, width=400e-9, height=600e-9)
        assert result == (400e-9, 600e-9, 50e-9)

    def test_returns_3tuple(self):
        result = _make_port_shape(axis=0, resolution=1.0, width=2.0, height=3.0)
        assert isinstance(result, tuple)
        assert len(result) == 3

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_resolution_at_propagation_axis(self, axis):
        result = _make_port_shape(axis=axis, resolution=123.0, width=456.0, height=789.0)
        assert result[axis] == 123.0

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_transverse_dims_are_width_and_height(self, axis):
        result = _make_port_shape(axis=axis, resolution=100.0, width=200.0, height=300.0)
        transverse = [i for i in range(3) if i != axis]
        assert result[transverse[0]] == 200.0
        assert result[transverse[1]] == 300.0

    def test_asymmetric_cross_section(self):
        result = _make_port_shape(axis=1, resolution=10.0, width=50.0, height=90.0)
        assert result == (50.0, 10.0, 90.0)
