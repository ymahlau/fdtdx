import jax
import jax.numpy as jnp

from fdtdx.core.physics.losses import metric_efficiency

# ──────────────────────────────────────────────────────────────
# metric_efficiency
# ──────────────────────────────────────────────────────────────


def _make_detector_states(values: dict[str, float], metric: str = "energy"):
    """Helper: build detector_states from {name: scalar_value}."""
    return {name: {metric: jnp.array(val)} for name, val in values.items()}


def test_single_pair():
    """Basic efficiency: equal input/output gives 1.0."""
    states = _make_detector_states({"src": 4.0, "det": 4.0})
    obj, info = metric_efficiency(states, ["src"], ["det"], "energy")
    assert jnp.allclose(obj, 1.0)
    assert jnp.allclose(info["det_by_src_efficiency"], 1.0)


def test_zero_input_returns_zero():
    """Efficiency is 0 when input is zero (division-by-zero protection)."""
    states = _make_detector_states({"src": 0.0, "det": 5.0})
    obj, info = metric_efficiency(states, ["src"], ["det"], "energy")
    assert jnp.allclose(obj, 0.0)
    assert jnp.allclose(info["det_by_src_efficiency"], 0.0)


def test_multiple_outputs():
    """Mean efficiency across two output detectors."""
    states = _make_detector_states({"src": 10.0, "d1": 5.0, "d2": 10.0})
    obj, info = metric_efficiency(states, ["src"], ["d1", "d2"], "energy")
    # efficiencies: 0.5, 1.0 → mean = 0.75
    assert jnp.allclose(obj, 0.75)
    assert jnp.allclose(info["d1_by_src_efficiency"], 0.5)
    assert jnp.allclose(info["d2_by_src_efficiency"], 1.0)


def test_multiple_inputs():
    """Mean efficiency across two input detectors."""
    states = _make_detector_states({"s1": 10.0, "s2": 5.0, "det": 5.0})
    obj, info = metric_efficiency(states, ["s1", "s2"], ["det"], "energy")
    # efficiencies: 5/10=0.5, 5/5=1.0 → mean = 0.75
    assert jnp.allclose(obj, 0.75)
    assert jnp.allclose(info["det_by_s1_efficiency"], 0.5)
    assert jnp.allclose(info["det_by_s2_efficiency"], 1.0)


def test_info_keys_multiple_pairs():
    """Info contains all expected keys for multiple in/out pairs."""
    states = _make_detector_states({"s1": 1.0, "s2": 1.0, "d1": 1.0, "d2": 1.0})
    _, info = metric_efficiency(states, ["s1", "s2"], ["d1", "d2"], "energy")
    expected_keys = {
        "s1_energy",
        "s2_energy",
        "d1_energy",
        "d2_energy",
        "d1_by_s1_efficiency",
        "d2_by_s1_efficiency",
        "d1_by_s2_efficiency",
        "d2_by_s2_efficiency",
    }
    assert expected_keys == set(info.keys())


def test_custom_metric_name():
    """Info keys use the provided metric name."""
    states = {"src": {"power": jnp.array(3.0)}, "det": {"power": jnp.array(1.5)}}
    _, info = metric_efficiency(states, ["src"], ["det"], "power")
    assert "src_power" in info
    assert "det_power" in info
    assert jnp.allclose(info["det_by_src_efficiency"], 0.5)


def test_array_values_are_meaned():
    """Detector states with arrays use .mean() for comparison."""
    states = {
        "src": {"energy": jnp.array([2.0, 4.0, 6.0])},  # mean=4
        "det": {"energy": jnp.array([1.0, 3.0, 2.0])},  # mean=2
    }
    obj, _ = metric_efficiency(states, ["src"], ["det"], "energy")
    assert jnp.allclose(obj, 0.5)


def test_input_stop_gradient():
    """Input values use stop_gradient (verified via jax.grad)."""

    def loss_fn_det(det_val):
        states = {
            "src": {"energy": jnp.array(4.0)},
            "det": {"energy": det_val},
        }
        obj, _ = metric_efficiency(states, ["src"], ["det"], "energy")
        return obj

    grad = jax.grad(loss_fn_det)(jnp.array(2.0))
    # gradient of (det/src) w.r.t. det = 1/src = 0.25
    assert jnp.allclose(grad, 0.25)

    def loss_fn_src(src_val):
        states = {
            "src": {"energy": src_val},
            "det": {"energy": jnp.array(2.0)},
        }
        obj, _ = metric_efficiency(states, ["src"], ["det"], "energy")
        return obj

    # stop_gradient applied to source values → gradient w.r.t. src must be 0
    grad_src = jax.grad(loss_fn_src)(jnp.array(4.0))
    assert jnp.allclose(grad_src, 0.0)
