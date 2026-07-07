"""Integration tests for sub-pixel (sub-cell) dielectric smoothing on StaticMultiMaterialObject.

Sub-pixel smoothing replaces the binary (staircased) voxel permittivity at material interfaces with an
anisotropic effective permittivity following Farjadpour et al. (Meep): the arithmetic mean of eps for the
field components tangential to the interface and the harmonic mean for the component normal to it. This
removes the first-order Yee staircasing error at strong dielectric jumps.

The scene here is a SiN half-plane in vacuum: the material footprint fills x < x_edge (with x_edge placed
at a sub-cell position) and spans the full y and z extent, so the only interface is the vertical x-normal
face at x_edge -- the in-plane fill-fraction case that GDS sidewall smoothing (issue #373) targets. For a
cell on that face with fill fraction f and normal n = x_hat the effective tensor is diag(eps_h, eps_bar,
eps_bar) with

    eps_bar = f * eps_SiN + (1 - f) * eps_vac        (arithmetic, tangential y and z)
    eps_h   = 1 / (f / eps_SiN + (1 - f) / eps_vac)  (harmonic, normal x)

The tests assert this relation exactly, plus the allocation logic (3-component diagonal vs 9-component full
tensor) and that disabling smoothing is a no-op.
"""

import gdstk
import jax
import numpy as np

from fdtdx.config import SimulationConfig
from fdtdx.core.grid import UniformGrid
from fdtdx.fdtd.initialization import place_objects
from fdtdx.materials import Material
from fdtdx.objects.static_material.gds_layer_stack import GDSLayerSpec, gds_layer_stack
from fdtdx.objects.static_material.static import SimulationVolume

EPS_SIN = 4.0
EPS_VAC = 1.0
SPACING = 1e-6  # 1 um cells
N = 8  # 8 x 8 x 8 um domain
X_EDGE = 0.3e-6  # material footprint boundary at a deliberately sub-cell x position


def _build_halfplane_arrays(*, subpixel: bool, full_tensor: bool = False):
    """Place a SiN half-plane (x < X_EDGE) spanning the full y/z extent and return the ArrayContainer.

    The footprint's only interior boundary is the vertical x-normal face at X_EDGE; the layer fills the
    whole z-extent (no internal horizontal face) and the footprint runs far past the domain in y (no
    internal y face), so every smoothed cell has a clean n = x_hat interface normal.
    """
    volume = SimulationVolume(name="vol", partial_grid_shape=(N, N, N), material=Material(permittivity=EPS_VAC))

    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("HALF")
    big = 1e3  # um; runs far past the domain in -x and +/-y so only the +x edge falls inside
    cell.add(gdstk.Polygon([(-big, -big), (X_EDGE * 1e6, -big), (X_EDGE * 1e6, big), (-big, big)], layer=1, datatype=0))

    layers = [
        GDSLayerSpec(
            gds_layer=1,
            material_name="SiN",
            thickness=N * SPACING,  # full z-extent -> no internal horizontal face
            z_base=0.0,
            sidewall_angle=90.0,
            subpixel_smoothing=subpixel,
            subpixel_full_tensor=full_tensor,
        )
    ]
    materials = {"SiN": Material(permittivity=EPS_SIN), "vac": Material(permittivity=EPS_VAC)}
    objs, constraints = gds_layer_stack(lib, "HALF", layers, materials, volume, gds_center=(0.0, 0.0))

    config = SimulationConfig(grid=UniformGrid(spacing=SPACING), time=100e-15, backend="cpu")
    _obj, arrays, _params, _cfg, _info = place_objects([volume, *objs], config, constraints, jax.random.PRNGKey(0))
    return arrays


def _forward_diag(ip):
    """Return (eps_xx, eps_yy, eps_zz) from an inv-permittivity array of 3 or 9 components."""
    if ip.shape[0] == 3:
        return 1.0 / ip[0], 1.0 / ip[1], 1.0 / ip[2]
    eps = np.linalg.inv(np.moveaxis(ip, 0, -1).reshape(*ip.shape[1:], 3, 3))
    return eps[..., 0, 0], eps[..., 1, 1], eps[..., 2, 2]


def test_subpixel_off_is_isotropic_and_unchanged():
    """Without smoothing the permittivity stays a 1-component isotropic array (binary staircase)."""
    ip = np.asarray(_build_halfplane_arrays(subpixel=False).inv_permittivities)
    assert ip.shape[0] == 1  # isotropic scalar allocation
    eps = 1.0 / ip[0]
    # every cell is either pure SiN or pure vacuum -- no intermediate (smoothed) values
    assert np.all(np.isclose(eps, EPS_SIN) | np.isclose(eps, EPS_VAC))


def test_diagonal_subpixel_allocation_and_bulk():
    """Diagonal smoothing (default) allocates a 3-component array and preserves the bulk material values."""
    ip = np.asarray(_build_halfplane_arrays(subpixel=True, full_tensor=False).inv_permittivities)
    assert ip.shape[0] == 3  # diagonal anisotropic allocation
    exx, eyy, ezz = _forward_diag(ip)
    for e in (exx, eyy, ezz):
        assert np.isclose(e.max(), EPS_SIN)  # a fully-SiN cell exists
        assert np.isclose(e.min(), EPS_VAC)  # a fully-vacuum cell exists


def test_diagonal_subpixel_farjadpour_relation_on_x_face():
    """The x-normal interface cell must obey the exact Farjadpour arithmetic/harmonic split.

    For a cell with fill fraction f and normal n = x_hat: eps_yy = eps_zz = arithmetic mean (tangential),
    eps_xx = harmonic mean (normal). We recover f from the tangential component and assert the normal one
    matches the harmonic mean -- a check that is exact and independent of the exact fill the mesh produced.
    """
    ip = np.asarray(_build_halfplane_arrays(subpixel=True, full_tensor=False).inv_permittivities)
    exx, eyy, ezz = _forward_diag(ip)

    aniso = np.abs(exx - eyy)
    idx = np.unravel_index(int(np.argmax(aniso)), aniso.shape)
    assert aniso[idx] > 1e-6, "expected at least one anisotropic (smoothed) interface cell"

    # x-normal interface: the two tangential components (y, z) are equal; the normal (x) is smaller.
    assert np.isclose(eyy[idx], ezz[idx], rtol=1e-6), "tangential components must match for an x-normal interface"
    assert exx[idx] < eyy[idx] - 1e-6, "normal (harmonic) component must be below the tangential (arithmetic) one"

    # recover the fill fraction from the arithmetic (tangential) component and verify the harmonic (normal) one
    f = (eyy[idx] - EPS_VAC) / (EPS_SIN - EPS_VAC)
    assert 0.0 < f < 1.0
    eps_h_expected = 1.0 / (f / EPS_SIN + (1.0 - f) / EPS_VAC)
    assert np.isclose(exx[idx], eps_h_expected, rtol=1e-6), (
        f"harmonic mean mismatch: eps_xx={exx[idx]:.6f} expected={eps_h_expected:.6f} (f={f:.4f})"
    )


def test_full_tensor_subpixel_allocation_and_diagonal_on_axis_aligned_face():
    """The full-tensor variant allocates 9 components; for an x-normal face it stays diagonal (no off-diag)."""
    ip = np.asarray(_build_halfplane_arrays(subpixel=True, full_tensor=True).inv_permittivities)
    assert ip.shape[0] == 9  # full tensor allocation

    eps = np.linalg.inv(np.moveaxis(ip, 0, -1).reshape(*ip.shape[1:], 3, 3))
    exx, eyy, ezz = eps[..., 0, 0], eps[..., 1, 1], eps[..., 2, 2]
    offdiag = np.abs(eps[..., 0, 1]) + np.abs(eps[..., 0, 2]) + np.abs(eps[..., 1, 2])

    idx = np.unravel_index(int(np.argmax(np.abs(exx - eyy))), exx.shape)
    assert np.isclose(eyy[idx], ezz[idx], rtol=1e-6)
    assert exx[idx] < eyy[idx] - 1e-6
    assert offdiag[idx] < 1e-6, "axis-aligned interface must have zero off-diagonal terms"
    f = (eyy[idx] - EPS_VAC) / (EPS_SIN - EPS_VAC)
    assert np.isclose(exx[idx], 1.0 / (f / EPS_SIN + (1.0 - f) / EPS_VAC), rtol=1e-6)


def test_diagonal_and_full_tensor_agree_on_diagonal():
    """For an axis-aligned interface the cheap diagonal path equals the diagonal of the full tensor."""
    diag = np.asarray(_build_halfplane_arrays(subpixel=True, full_tensor=False).inv_permittivities)
    full = np.asarray(_build_halfplane_arrays(subpixel=True, full_tensor=True).inv_permittivities)
    # diagonal components of the full 9-tensor (indices 0, 4, 8) must match the 3-component diagonal array
    assert np.allclose(diag[0], full[0], rtol=1e-6)  # xx
    assert np.allclose(diag[1], full[4], rtol=1e-6)  # yy
    assert np.allclose(diag[2], full[8], rtol=1e-6)  # zz


# ---------------------------------------------------------------------------
# Tilted sidewall (issue #373): the trapezoidal wall produces a non-axis-aligned
# interface normal in the x-z plane -> genuine off-diagonal (eps_xz) tensor terms.
# ---------------------------------------------------------------------------


def _build_wedge_arrays(*, full_tensor: bool, sidewall_angle: float = 50.0):
    """SiN half-plane with a steep sidewall: the x-edge slants across z (a wall in the x-z plane).

    z_base=1 um, thickness=6 um, 50 deg wall -> the footprint boundary sweeps ~5 um in x over the 6 um of
    height, i.e. a slanted interface whose normal lies in the x-z plane. The lateral taper is far above one
    cell, so the per-z-slice trapezoidal fill path engages. The edge x (2.7 um) and angle are chosen so the
    boundary lands mid-cell at each height (a sub-cell fractional fill, not a cell-aligned staircase).
    """
    volume = SimulationVolume(name="vol", partial_grid_shape=(N, N, N), material=Material(permittivity=EPS_VAC))
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("WEDGE")
    big = 1e3
    x_edge_um = 2.7  # bottom edge at x = 2.7 um; erodes upward over the ~5 um wall (mid-cell at each z)
    cell.add(gdstk.Polygon([(-big, -big), (x_edge_um, -big), (x_edge_um, big), (-big, big)], layer=1, datatype=0))
    layers = [
        GDSLayerSpec(
            gds_layer=1,
            material_name="SiN",
            thickness=6e-6,
            z_base=1e-6,
            sidewall_angle=sidewall_angle,
            reference_plane="bottom",
            subpixel_smoothing=True,
            subpixel_full_tensor=full_tensor,
        )
    ]
    materials = {"SiN": Material(permittivity=EPS_SIN), "vac": Material(permittivity=EPS_VAC)}
    objs, constraints = gds_layer_stack(lib, "WEDGE", layers, materials, volume, gds_center=(0.0, 0.0))
    config = SimulationConfig(grid=UniformGrid(spacing=SPACING), time=100e-15, backend="cpu")
    _obj, arrays, _params, _cfg, _info = place_objects([volume, *objs], config, constraints, jax.random.PRNGKey(0))
    return arrays


def test_tilted_sidewall_produces_offdiagonal_tensor():
    """A slanted (non-axis-aligned) sidewall must yield genuine off-diagonal eps_xz terms (issue #373).

    This is the payoff of the full-tensor path: for an interface normal in the x-z plane the Farjadpour
    tensor has eps_xz = -(eps_bar - eps_h) * n_x * n_z != 0, which the diagonal path cannot represent.
    """
    ip = np.asarray(_build_wedge_arrays(full_tensor=True).inv_permittivities)
    assert ip.shape[0] == 9
    eps = np.linalg.inv(np.moveaxis(ip, 0, -1).reshape(*ip.shape[1:], 3, 3))
    exz = eps[..., 0, 2]
    # the slanted wall spans x and z -> a substantial x-z coupling must appear somewhere
    assert np.max(np.abs(exz)) > 0.1, (
        f"expected off-diagonal eps_xz from the tilted wall, got {np.max(np.abs(exz)):.3e}"
    )
    # the y-z and x-y couplings stay ~zero (the wall is invariant in y)
    assert np.max(np.abs(eps[..., 0, 1])) < 1e-6  # eps_xy
    assert np.max(np.abs(eps[..., 1, 2])) < 1e-6  # eps_yz


def test_tilted_sidewall_smoothing_tracks_the_wall_across_z():
    """The smoothed (interface) cells should follow the slanted wall: their x-position shifts with z."""
    ip = np.asarray(_build_wedge_arrays(full_tensor=False).inv_permittivities)
    exx, eyy, _ezz = _forward_diag(ip)
    smoothed = np.abs(exx - eyy) > 1e-3  # cells with tangential/normal contrast (on the interface)
    # for the lowest and highest interior z-slices, the mean x-index of smoothed cells must differ
    # (the wall moved in x) -- a direct check that the fill fraction is z-dependent, not separable.
    zs = np.where(smoothed.any(axis=(0, 1)))[0]
    assert zs.size >= 2, "expected smoothed cells across multiple z-slices"
    xs_low = np.where(smoothed[:, :, zs[0]].any(axis=1))[0].mean()
    xs_high = np.where(smoothed[:, :, zs[-1]].any(axis=1))[0].mean()
    assert abs(xs_high - xs_low) > 1.0, (
        f"interface x-position should shift with z for a tilted wall (low={xs_low:.1f}, high={xs_high:.1f})"
    )
