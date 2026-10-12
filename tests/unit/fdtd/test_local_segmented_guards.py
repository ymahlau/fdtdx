import jax
import jax.numpy as jnp
import pytest

import fdtdx
from fdtdx.config import GradientConfig, SimulationConfig
from fdtdx.constants import c as c0
from fdtdx.core.grid import UniformGrid
from fdtdx.materials import Material
from fdtdx.objects.detectors.poynting_flux import PoyntingFluxDetector
from fdtdx.objects.device.device import Device
from fdtdx.objects.object import GridCoordinateConstraint

_RESOLUTION = 50e-9
_SIM_TIME = 10e-15
_PML_CELLS = 3
_VOLUME_CELLS = 14


def _build_base_scene(
    gradient_config=None,
    with_device=True,
    dev_coords=(5, 5, 5),
    src_coords=(3, 3, 3),
    det_coords=(5, 5, 10),
    lossy_device=False,
    detector_type="phasor",
):
    if gradient_config is None:
        gradient_config = GradientConfig(
            method="reversible",
            recording_mode="local_segmented",
            num_checkpoints_reversible=1,
        )

    config = SimulationConfig(
        time=_SIM_TIME,
        grid=UniformGrid(spacing=_RESOLUTION),
        backend="cpu",
        dtype=jnp.float32,
        gradient_config=gradient_config,
    )
    n = _VOLUME_CELLS
    objects, constraints = [], []
    volume = fdtdx.SimulationVolume(partial_grid_shape=(n, n, n))
    objects.append(volume)
    bound_dict, c_list = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML_CELLS), volume
    )
    objects.extend(bound_dict.values())
    constraints.extend(c_list)

    if with_device:
        if lossy_device:
            materials = {
                "air": Material(permittivity=1.0),
                "si": Material(permittivity=2.25, electric_conductivity=1e3),
            }
        else:
            materials = {
                "air": Material(permittivity=1.0),
                "si": Material(permittivity=2.25),
            }
        dev = Device(
            name="dev",
            partial_grid_shape=(4, 4, 4),
            partial_voxel_grid_shape=(1, 1, 1),
            materials=materials,
            param_transforms=[],
        )
        c_dev = GridCoordinateConstraint(
            object="dev", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=list(dev_coords)
        )
        objects.append(dev)
        constraints.append(c_dev)

    wave = fdtdx.WaveCharacter(frequency=c0 / 800e-9)
    source = fdtdx.PointDipoleSource(
        name="dip", partial_grid_shape=(1, 1, 1), wave_character=wave, polarization=0, amplitude=1.0
    )
    c_source = GridCoordinateConstraint(
        object="dip", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=list(src_coords)
    )
    objects.append(source)
    constraints.append(c_source)

    if detector_type == "phasor":
        det = fdtdx.PhasorDetector(
            name="det", partial_grid_shape=(4, 4, 1), wave_characters=(wave,), components=("Ex",), plot=False
        )
    elif detector_type == "field":
        det = fdtdx.FieldDetector(name="det", partial_grid_shape=(4, 4, 1), components=("Ex",), plot=False)
    elif detector_type == "poynting":
        det = PoyntingFluxDetector(name="det", partial_grid_shape=(4, 4, 1), plot=False)
    else:
        raise ValueError(f"Unknown detector_type: {detector_type}")

    c_det = GridCoordinateConstraint(object="det", axes=[0, 1, 2], sides=["-", "-", "-"], coordinates=list(det_coords))
    objects.append(det)
    constraints.append(c_det)

    key = jax.random.PRNGKey(0)
    obj, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, obj, _ = fdtdx.apply_params(arrays, obj, params, key)
    return obj, arrays, config, key


class TestLocalSegmentedGuards:
    def test_reject_lossy_medium(self):
        with pytest.raises(ValueError, match="lossless materials"):
            _build_base_scene(lossy_device=True)

    def test_reject_non_linear_detector(self):
        with pytest.raises(ValueError, match="linear detectors"):
            _build_base_scene(detector_type="poynting")

    def test_reject_missing_devices(self):
        with pytest.raises(ValueError, match="requires at least one optimizable device"):
            _build_base_scene(with_device=False)

    def test_reject_boundary_margin_violation(self):
        # dev at (0, 5, 5) violates 1-cell shell margin
        with pytest.raises(ValueError, match="one-cell shell"):
            _build_base_scene(dev_coords=(0, 5, 5))

    def test_reject_pml_intersection(self):
        # PML is at [0, 3) and [11, 14). Dev at (3, 5, 5) has 1-cell shell at [2, 7) which intersects PML [0, 3)
        with pytest.raises(ValueError, match="intersects PerfectlyMatchedLayer"):
            _build_base_scene(dev_coords=(3, 5, 5))

    def test_reject_source_intersection(self):
        # Source at (4, 5, 5) intersects dev 1-cell shell [4, 10)
        with pytest.raises(ValueError, match="intersects PointDipoleSource"):
            _build_base_scene(dev_coords=(5, 5, 5), src_coords=(4, 5, 5))

    def test_reject_split_adjoint_boundary_detector(self):
        # Detector at z=13 touches domain boundary (z_max = 13 for 14-cell grid)
        grad_cfg = GradientConfig(
            method="reversible",
            recording_mode="local_segmented",
            num_checkpoints_reversible=1,
            split_detector_adjoint=True,
        )
        with pytest.raises(ValueError, match="split_detector_adjoint requires interior detectors"):
            _build_base_scene(gradient_config=grad_cfg, det_coords=(5, 5, 13))

    def test_reject_dispersive_medium(self):
        obj, arrays, config, key = _build_base_scene()
        # Simulate dispersive array presence
        arrays_disp = arrays.aset("dispersive_c1", jnp.ones((1, 1, 1, 1, 1)))
        with pytest.raises(ValueError, match="does not support dispersive materials"):
            fdtdx.run_fdtd(arrays_disp, obj, config, key)

    def test_reject_non_diagonal_tensor(self):
        obj, arrays, config, key = _build_base_scene()
        # 9 components represents full anisotropic tensor
        shape = (9, *arrays.inv_permittivities.shape[1:])
        arrays_aniso = arrays.aset("inv_permittivities", jnp.ones(shape))
        with pytest.raises(ValueError, match=r"only supports real isotropic.*or diagonal anisotropic"):
            fdtdx.run_fdtd(arrays_aniso, obj, config, key)

    def test_accept_field_detector_split_adjoint(self):
        # FieldDetector with split_detector_adjoint=True must be accepted by guards
        grad_cfg = GradientConfig(
            method="reversible",
            recording_mode="local_segmented",
            split_detector_adjoint=True,
        )
        obj, _arrays, _config, _key = _build_base_scene(gradient_config=grad_cfg, detector_type="field")
        assert len(obj.forward_detectors) == 1
        assert isinstance(obj.forward_detectors[0], fdtdx.FieldDetector)

    def test_accept_default_construction_local_segmented(self):
        # Default GradientConfig with recording_mode="local_segmented" (num_checkpoints_reversible=0)
        grad_cfg = GradientConfig(
            method="reversible",
            recording_mode="local_segmented",
        )
        _obj, _arrays, config, _key = _build_base_scene(gradient_config=grad_cfg)
        assert config.gradient_config.num_checkpoints_reversible == 0
