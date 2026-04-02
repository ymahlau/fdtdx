"""Unit tests for objects/boundaries/initialization.py.

Tests BoundaryConfig methods and boundary_objects_from_config factory function.
"""

import pytest

from fdtdx.objects.boundaries.bloch import BlochBoundary
from fdtdx.objects.boundaries.initialization import (
    BoundaryConfig,
    boundary_objects_from_config,
)
from fdtdx.objects.boundaries.perfectly_matched_layer import PerfectlyMatchedLayer
from fdtdx.objects.static_material.static import SimulationVolume


@pytest.fixture
def default_config():
    return BoundaryConfig()


@pytest.fixture
def volume():
    return SimulationVolume(partial_grid_shape=(100, 100, 100))


class TestBoundaryConfigDefaults:
    """Tests for BoundaryConfig default values."""

    def test_all_boundaries_pml_by_default(self, default_config):
        type_dict = default_config.get_type_dict()
        assert all(v == "pml" for v in type_dict.values())

    def test_default_thickness_is_10(self, default_config):
        d = default_config.get_dict()
        assert all(v == 10 for v in d.values())

    def test_kappa_none_by_default(self, default_config):
        for prop in ("kappa_start", "kappa_end"):
            d = default_config.get_kappa_dict(prop)
            assert all(v is None for v in d.values())

    def test_alpha_none_by_default(self, default_config):
        for prop in ("alpha_start", "alpha_end"):
            d = default_config.get_alpha_dict(prop)
            assert all(v is None for v in d.values())

    def test_sigma_none_by_default(self, default_config):
        for prop in ("sigma_start", "sigma_end"):
            d = default_config.get_sigma_dict(prop)
            assert all(v is None for v in d.values())

    def test_orders_none_by_default(self, default_config):
        for prop in ("sigma_order", "alpha_order", "kappa_order"):
            d = default_config.get_order_dict(prop)
            assert all(v is None for v in d.values())


class TestBoundaryConfigGetDict:
    """Tests for BoundaryConfig.get_dict()."""

    def test_returns_six_boundaries(self):
        cfg = BoundaryConfig()
        d = cfg.get_dict()
        assert set(d.keys()) == {"min_x", "max_x", "min_y", "max_y", "min_z", "max_z"}

    def test_custom_thickness_per_boundary(self):
        cfg = BoundaryConfig(
            thickness_grid_minx=5,
            thickness_grid_maxx=15,
            thickness_grid_miny=8,
            thickness_grid_maxy=12,
            thickness_grid_minz=6,
            thickness_grid_maxz=20,
        )
        d = cfg.get_dict()
        assert d["min_x"] == 5
        assert d["max_x"] == 15
        assert d["min_y"] == 8
        assert d["max_y"] == 12
        assert d["min_z"] == 6
        assert d["max_z"] == 20


class TestBoundaryConfigGetTypeDict:
    """Tests for BoundaryConfig.get_type_dict()."""

    def test_returns_six_boundaries(self):
        cfg = BoundaryConfig()
        d = cfg.get_type_dict()
        assert set(d.keys()) == {"min_x", "max_x", "min_y", "max_y", "min_z", "max_z"}

    def test_mixed_types(self):
        cfg = BoundaryConfig(
            boundary_type_minx="periodic",
            boundary_type_maxx="periodic",
            boundary_type_miny="pml",
            boundary_type_maxy="pml",
            boundary_type_minz="pml",
            boundary_type_maxz="pml",
        )
        d = cfg.get_type_dict()
        assert d["min_x"] == "periodic"
        assert d["max_x"] == "periodic"
        assert d["min_y"] == "pml"


class TestBoundaryConfigGetKappaDict:
    """Tests for BoundaryConfig.get_kappa_dict()."""

    def test_kappa_start(self):
        cfg = BoundaryConfig(kappa_start_minx=1.0, kappa_start_maxy=2.0)
        d = cfg.get_kappa_dict("kappa_start")
        assert d["min_x"] == 1.0
        assert d["max_y"] == 2.0
        assert d["max_x"] is None

    def test_kappa_end(self):
        cfg = BoundaryConfig(kappa_end_minz=3.0)
        d = cfg.get_kappa_dict("kappa_end")
        assert d["min_z"] == 3.0
        assert d["max_z"] is None

    def test_invalid_prop_raises(self):
        cfg = BoundaryConfig()
        with pytest.raises(Exception):
            cfg.get_kappa_dict("kappa_middle")  # type: ignore


class TestBoundaryConfigGetAlphaDict:
    """Tests for BoundaryConfig.get_alpha_dict()."""

    def test_alpha_start(self):
        cfg = BoundaryConfig(alpha_start_minx=0.01, alpha_start_maxz=0.05)
        d = cfg.get_alpha_dict("alpha_start")
        assert d["min_x"] == 0.01
        assert d["max_z"] == 0.05
        assert d["min_y"] is None

    def test_alpha_end(self):
        cfg = BoundaryConfig(alpha_end_maxx=0.0)
        d = cfg.get_alpha_dict("alpha_end")
        assert d["max_x"] == 0.0

    def test_invalid_prop_raises(self):
        cfg = BoundaryConfig()
        with pytest.raises(Exception):
            cfg.get_alpha_dict("alpha_mid")  # type: ignore


class TestBoundaryConfigGetSigmaDict:
    """Tests for BoundaryConfig.get_sigma_dict()."""

    def test_sigma_start(self):
        cfg = BoundaryConfig(sigma_start_minx=0.0, sigma_start_miny=0.5)
        d = cfg.get_sigma_dict("sigma_start")
        assert d["min_x"] == 0.0
        assert d["min_y"] == 0.5

    def test_sigma_end(self):
        cfg = BoundaryConfig(sigma_end_maxz=1e6)
        d = cfg.get_sigma_dict("sigma_end")
        assert d["max_z"] == 1e6

    def test_invalid_prop_raises(self):
        cfg = BoundaryConfig()
        with pytest.raises(Exception):
            cfg.get_sigma_dict("sigma_peak")  # type: ignore


class TestBoundaryConfigGetOrderDict:
    """Tests for BoundaryConfig.get_order_dict()."""

    def test_sigma_order(self):
        cfg = BoundaryConfig(sigma_order_minx=3.0, sigma_order_maxy=2.0)
        d = cfg.get_order_dict("sigma_order")
        assert d["min_x"] == 3.0
        assert d["max_y"] == 2.0

    def test_alpha_order(self):
        cfg = BoundaryConfig(alpha_order_minz=1.0)
        d = cfg.get_order_dict("alpha_order")
        assert d["min_z"] == 1.0

    def test_kappa_order(self):
        cfg = BoundaryConfig(kappa_order_maxx=4.0)
        d = cfg.get_order_dict("kappa_order")
        assert d["max_x"] == 4.0

    def test_invalid_prop_raises(self):
        cfg = BoundaryConfig()
        with pytest.raises(Exception):
            cfg.get_order_dict("sigma_exponent")  # type: ignore


class TestBoundaryConfigInsideBoundarySlice:
    """Tests for BoundaryConfig.get_inside_boundary_slice()."""

    def test_all_pml_excludes_boundaries(self):
        cfg = BoundaryConfig(
            thickness_grid_minx=5,
            thickness_grid_maxx=5,
            thickness_grid_miny=8,
            thickness_grid_maxy=8,
            thickness_grid_minz=10,
            thickness_grid_maxz=10,
        )
        sx, sy, sz = cfg.get_inside_boundary_slice()
        assert sx == slice(6, -6)
        assert sy == slice(9, -9)
        assert sz == slice(11, -11)

    def test_periodic_boundary_uses_zero_start(self):
        cfg = BoundaryConfig(
            boundary_type_minx="periodic",
            boundary_type_maxx="periodic",
            boundary_type_miny="pml",
            boundary_type_maxy="pml",
            boundary_type_minz="pml",
            boundary_type_maxz="pml",
            thickness_grid_miny=5,
            thickness_grid_maxy=5,
            thickness_grid_minz=5,
            thickness_grid_maxz=5,
        )
        sx, sy, sz = cfg.get_inside_boundary_slice()
        # periodic x: starts at 0, ends at None
        assert sx.start == 0
        assert sx.stop is None

    def test_all_periodic_is_full_slice(self):
        cfg = BoundaryConfig(
            boundary_type_minx="periodic",
            boundary_type_maxx="periodic",
            boundary_type_miny="periodic",
            boundary_type_maxy="periodic",
            boundary_type_minz="periodic",
            boundary_type_maxz="periodic",
        )
        sx, sy, sz = cfg.get_inside_boundary_slice()
        assert sx == slice(0, None)
        assert sy == slice(0, None)
        assert sz == slice(0, None)


class TestBoundaryConfigFromUniformBound:
    """Tests for BoundaryConfig.from_uniform_bound()."""

    def test_default_thickness(self):
        cfg = BoundaryConfig.from_uniform_bound(thickness=10)
        d = cfg.get_dict()
        assert all(v == 10 for v in d.values())

    def test_custom_thickness(self):
        cfg = BoundaryConfig.from_uniform_bound(thickness=20)
        d = cfg.get_dict()
        assert all(v == 20 for v in d.values())

    def test_default_type_is_pml(self):
        cfg = BoundaryConfig.from_uniform_bound()
        d = cfg.get_type_dict()
        assert all(v == "pml" for v in d.values())

    def test_periodic_type(self):
        cfg = BoundaryConfig.from_uniform_bound(boundary_type="periodic")
        d = cfg.get_type_dict()
        assert all(v == "periodic" for v in d.values())

    def test_override_types(self):
        cfg = BoundaryConfig.from_uniform_bound(
            boundary_type="pml",
            override_types={"min_x": "periodic", "max_x": "periodic"},
        )
        d = cfg.get_type_dict()
        assert d["min_x"] == "periodic"
        assert d["max_x"] == "periodic"
        assert d["min_y"] == "pml"
        assert d["max_z"] == "pml"

    def test_kappa_values_propagated(self):
        cfg = BoundaryConfig.from_uniform_bound(kappa_start=1.0, kappa_end=2.0)
        starts = cfg.get_kappa_dict("kappa_start")
        ends = cfg.get_kappa_dict("kappa_end")
        assert all(v == 1.0 for v in starts.values())
        assert all(v == 2.0 for v in ends.values())

    def test_sigma_values_propagated(self):
        cfg = BoundaryConfig.from_uniform_bound(sigma_start=0.0, sigma_end=1e6)
        starts = cfg.get_sigma_dict("sigma_start")
        ends = cfg.get_sigma_dict("sigma_end")
        assert all(v == 0.0 for v in starts.values())
        assert all(v == 1e6 for v in ends.values())

    def test_alpha_values_propagated(self):
        cfg = BoundaryConfig.from_uniform_bound(alpha_start=0.01, alpha_end=0.0)
        starts = cfg.get_alpha_dict("alpha_start")
        ends = cfg.get_alpha_dict("alpha_end")
        assert all(v == 0.01 for v in starts.values())
        assert all(v == 0.0 for v in ends.values())

    def test_order_values_propagated(self):
        cfg = BoundaryConfig.from_uniform_bound(sigma_order=3.0, kappa_order=1.0)
        sigma_orders = cfg.get_order_dict("sigma_order")
        kappa_orders = cfg.get_order_dict("kappa_order")
        assert all(v == 3.0 for v in sigma_orders.values())
        assert all(v == 1.0 for v in kappa_orders.values())

    def test_none_override_types_defaults_to_empty(self):
        cfg = BoundaryConfig.from_uniform_bound(override_types=None)
        d = cfg.get_type_dict()
        assert all(v == "pml" for v in d.values())


class TestBoundaryObjectsFromConfig:
    """Tests for boundary_objects_from_config factory function."""

    def test_creates_six_boundaries_all_pml(self, volume):
        cfg = BoundaryConfig.from_uniform_bound(thickness=10)
        boundaries, constraints = boundary_objects_from_config(cfg, volume)
        assert len(boundaries) == 6
        assert set(boundaries.keys()) == {"min_x", "max_x", "min_y", "max_y", "min_z", "max_z"}

    def test_pml_type_objects(self, volume):
        cfg = BoundaryConfig.from_uniform_bound(boundary_type="pml")
        boundaries, _ = boundary_objects_from_config(cfg, volume)
        for b in boundaries.values():
            assert isinstance(b, PerfectlyMatchedLayer)

    def test_periodic_type_objects(self, volume):
        cfg = BoundaryConfig.from_uniform_bound(boundary_type="periodic")
        boundaries, _ = boundary_objects_from_config(cfg, volume)
        for b in boundaries.values():
            assert isinstance(b, BlochBoundary)
            assert b.bloch_vector == (0.0, 0.0, 0.0)

    def test_six_constraints_created(self, volume):
        cfg = BoundaryConfig.from_uniform_bound()
        _, constraints = boundary_objects_from_config(cfg, volume)
        assert len(constraints) == 6

    def test_mixed_types(self, volume):
        cfg = BoundaryConfig.from_uniform_bound(
            boundary_type="pml",
            override_types={"min_x": "periodic", "max_x": "periodic"},
        )
        boundaries, _ = boundary_objects_from_config(cfg, volume)
        assert isinstance(boundaries["min_x"], BlochBoundary)
        assert isinstance(boundaries["max_x"], BlochBoundary)
        assert isinstance(boundaries["min_y"], PerfectlyMatchedLayer)

    def test_unknown_type_raises(self, volume):
        """An unrecognised boundary_type string should raise ValueError."""
        cfg = BoundaryConfig(
            boundary_type_minx="unknown",
            boundary_type_maxx="pml",
            boundary_type_miny="pml",
            boundary_type_maxy="pml",
            boundary_type_minz="pml",
            boundary_type_maxz="pml",
        )
        with pytest.raises(ValueError, match="Unknown boundary type"):
            boundary_objects_from_config(cfg, volume)

    def test_constraint_references_volume(self, volume):
        cfg = BoundaryConfig.from_uniform_bound()
        _, constraints = boundary_objects_from_config(cfg, volume)
        for c in constraints:
            assert c.other_object == volume.name

    def test_pml_axis_and_direction(self, volume):
        cfg = BoundaryConfig.from_uniform_bound()
        boundaries, _ = boundary_objects_from_config(cfg, volume)
        assert boundaries["min_x"].axis == 0
        assert boundaries["min_x"].direction == "-"
        assert boundaries["max_x"].axis == 0
        assert boundaries["max_x"].direction == "+"
        assert boundaries["min_y"].axis == 1
        assert boundaries["max_z"].axis == 2
        assert boundaries["max_z"].direction == "+"
