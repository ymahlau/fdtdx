"""Validate the n_eff of fdtdx's sidewall extrusion against Tidy3D PolySlab (mode solver, local, 0 FC).

Builds a SiN/SiO2 waveguide cross-section with fdtdx's GDSLayerObject(sidewall_angle=...), extracts the
3-D voxel mask from fdtdx itself, solves its modes in Tidy3D, and compares n_eff(TE) to the equivalent
PolySlab. Confirms the geometry validation (vs PolySlab.inside) carries through to the physical mode index.
Run: python examples/validate_sidewall_neff.py  (needs tidy3d; ~1 min)."""
import numpy as np, jax, jax.numpy as jnp, tidy3d as td
from tidy3d.plugins.mode import ModeSolver
from fdtdx.config import SimulationConfig
from fdtdx.core.grid import UniformGrid
from fdtdx.materials import Material
from fdtdx.objects.static_material.gds_layer_stack import GDSLayerObject

wvl = 0.810; freq0 = td.C_0 / wvl
EPS_SIN, EPS_SIO2 = 4.0, 2.1         # generic SiN / SiO2 (nominal literature values)
W, H = 1.0e-6, 0.7e-6                # a representative SiN strip-waveguide cross-section (m)
RES = 10e-9
NX, NY, NZ = 24, 500, 400            # x (short, uniform guide), y (5um), z (4um)
materials = {"sio2": Material(permittivity=EPS_SIO2), "sin": Material(permittivity=EPS_SIN)}
config = SimulationConfig(time=1e-15, grid=UniformGrid(spacing=RES), backend="cpu",
                          dtype=jnp.float32, gradient_config=None)


def fdtdx_mask(sidewall_deg):
    """3-D SiN mask (NX,NY,NZ) from fdtdx's sidewall extrusion: the core slab embedded in SiO2 cladding."""
    Lx = NX * RES
    poly = np.array([[-Lx, -W / 2], [Lx, -W / 2], [Lx, W / 2], [-Lx, W / 2]])
    ncore = int(round(H / RES))
    obj = GDSLayerObject(materials=materials, polygons=[poly], gds_center=(0.0, 0.0),
                         material_name="sin", axis=2, thickness=H,
                         sidewall_angle=float(sidewall_deg), reference_plane="bottom")  # degrees, 90=vertical
    obj = obj.place_on_grid(grid_slice_tuple=((0, NX), (0, NY), (0, ncore)),
                            config=config, key=jax.random.PRNGKey(0))
    core = np.asarray(obj.get_voxel_mask_for_shape())     # (NX, NY, ncore) = the core slab itself
    full = np.zeros((NX, NY, NZ), dtype=bool)             # embed in z with SiO2 cladding above/below
    z0 = NZ // 2 - ncore // 2
    full[:, :, z0:z0 + ncore] = core
    return full


def neff_te(eps3d):
    """3 TE n_eff from a (nx,ny,nz) eps, via Tidy3D CustomMedium + mode solver."""
    nx, ny, nz = eps3d.shape
    ys = (np.arange(ny) - ny / 2 + 0.5) * RES * 1e6
    zs = (np.arange(nz) - nz / 2 + 0.5) * RES * 1e6
    eps2d = eps3d[nx // 2]                     # transverse y-z slice
    da = td.SpatialDataArray(eps2d[None], coords=dict(x=[0.0], y=ys, z=zs))
    box = td.Structure(geometry=td.Box(center=(0, 0, 0), size=(td.inf, td.inf, td.inf)),
                       medium=td.CustomMedium(permittivity=da))
    sim = td.Simulation(size=(1, ny * RES * 1e6, nz * RES * 1e6), structures=[box],
                        grid_spec=td.GridSpec.auto(min_steps_per_wvl=20, wavelength=0.81), run_time=1e-13,
                        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()), sources=[], monitors=[])
    ms = ModeSolver(simulation=sim, plane=td.Box(center=(0, 0, 0), size=(0, ny * RES * 1e6, nz * RES * 1e6)),
                    mode_spec=td.ModeSpec(num_modes=8), freqs=[freq0])
    d = ms.solve()
    te = [float(np.real(d.n_eff.isel(f=0).sel(mode_index=i).values))
          for i in range(8) if float(d.pol_fraction.te.isel(f=0).sel(mode_index=i).values) > 0.9]
    return sorted(te, reverse=True)[:3]


def polyslab_neff(sidewall_deg):
    sw = np.deg2rad(90.0 - sidewall_deg)
    wg = td.Structure(geometry=td.PolySlab(vertices=[(-50, -W / 2e-6), (50, -W / 2e-6), (50, W / 2e-6), (-50, W / 2e-6)],
                      slab_bounds=(-H / 2e-6, H / 2e-6), axis=2, sidewall_angle=sw, reference_plane="bottom"),
                      medium=td.Medium(permittivity=EPS_SIN))
    sim = td.Simulation(size=(1, 5, 4), structures=[wg], medium=td.Medium(permittivity=EPS_SIO2),
                        grid_spec=td.GridSpec.auto(min_steps_per_wvl=20, wavelength=0.81), run_time=1e-13,
                        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()), sources=[], monitors=[])
    ms = ModeSolver(simulation=sim, plane=td.Box(center=(0, 0, 0), size=(0, 5, 4)),
                    mode_spec=td.ModeSpec(num_modes=5, filter_pol="te"), freqs=[freq0])
    d = ms.solve()
    return [float(np.real(d.n_eff.isel(f=0).sel(mode_index=i).values)) for i in range(3)]


for deg in [90, 88, 86]:
    m = fdtdx_mask(deg)
    eps = np.where(m, EPS_SIN, EPS_SIO2)
    nf = neff_te(eps); nps = polyslab_neff(deg)
    dmax = max(abs(a - b) for a, b in zip(nf, nps))
    print(f"{deg}deg  fdtdx={[f'{x:.4f}' for x in nf]}  polyslab={[f'{x:.4f}' for x in nps]}  max|Δ|={dmax:.4f}", flush=True)
