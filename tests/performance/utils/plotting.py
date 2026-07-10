from __future__ import annotations

from pathlib import Path


def _overlay_gds(ax, gds_path, gds_layer: tuple[int, int], color: str = "#4C8BE0", alpha: float = 0.35) -> None:
    """Overlay GDS polygons (in µm) onto a matplotlib axes. Silently skips if gdstk is missing."""
    try:
        import gdstk
    except ImportError:
        return
    lib = gdstk.read_gds(str(gds_path))
    layer, datatype = gds_layer
    real_cells = [c for c in lib.cells if not c.name.startswith("$$$")]
    referenced = {ref.cell.name for c in real_cells for ref in c.references if hasattr(ref.cell, "name")}
    tops = [c for c in real_cells if c.name not in referenced] or real_cells
    for top in tops:
        for poly in top.get_polygons(layer=layer, datatype=datatype):
            pts = poly.points  # gdstk units are µm for gdsfactory GDS files
            ax.fill(pts[:, 0], pts[:, 1], facecolor=color, edgecolor="none", alpha=alpha, zorder=1)


def plot_field_intensity(
    detector_states: dict,
    config,
    figs_dir: Path,
    *,
    gds_path=None,
    gds_layer: tuple[int, int] = (1, 0),
    gds_center: tuple[float, float] = (0.0, 0.0),
    domain_shape: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_steps: int = 0,
    dx_nm: int = 50,
    y_lim: tuple[float, float] | None = None,
    det_info: dict | None = None,
) -> None:
    """Save inferno |Ey|² intensity plots for phasor detectors.

    Args:
        detector_states: ``final_arrays.detector_states`` dict from :func:`fdtdx.run_fdtd`.
        config: :class:`fdtdx.SimulationConfig` holding the grid edges.
        figs_dir: Output directory for PNG files.
        gds_path: Path to ``.gds`` file for polygon overlay. Skipped if ``None``.
        gds_layer: ``(layer, datatype)`` pair for GDS polygon overlay.
        gds_center: GDS (x, y) in metres mapped to the simulation centre.
        domain_shape: (x, y, z) total domain in metres.
        n_steps: Total simulation steps (for plot title).
        dx_nm: Grid spacing in nm (for plot title).
        y_lim: Optional (y_min, y_max) in GDS µm to crop the XY plot.
        det_info: Override axis metadata per detector.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    figs_dir = Path(figs_dir)
    figs_dir.mkdir(exist_ok=True)

    domain_x_m, domain_y_m, _ = domain_shape
    gds_cx_um = gds_center[0] * 1e6
    gds_cy_um = gds_center[1] * 1e6

    x_edges_um = np.array(config.grid.x_edges) * 1e6 - domain_x_m * 1e6 / 2 + gds_cx_um
    y_edges_um = np.array(config.grid.y_edges) * 1e6 - domain_y_m * 1e6 / 2 + gds_cy_um

    xc = 0.5 * (x_edges_um[:-1] + x_edges_um[1:])
    yc = 0.5 * (y_edges_um[:-1] + y_edges_um[1:])

    default_info: dict = {
        "phasor_xy": (xc, yc, y_lim),
    }
    info = det_info if det_info is not None else default_info

    for det_name, state in detector_states.items():
        if "phasor" not in state or det_name not in info:
            continue
        xvals, yvals, ylim = info[det_name]
        phasor = np.array(state["phasor"])[0, 0, 0].squeeze()
        intensity = np.abs(phasor) ** 2

        x_span = xvals[-1] - xvals[0]
        y_span = (ylim[1] - ylim[0]) if ylim is not None else (yvals[-1] - yvals[0])
        fig_w = 10.0
        fig_h = max(1.5, fig_w * y_span / x_span)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        ax.pcolormesh(xvals, yvals, intensity.T, cmap="turbo", vmin=0, vmax=intensity.max(), shading="nearest")
        ax.set_xlabel("x (µm)", fontsize=11)
        ax.set_ylabel("y (µm)", fontsize=11)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_title(f"fdtdx FDTD  (dx={dx_nm} nm, {n_steps} steps)", fontsize=11)

        out = figs_dir / f"{det_name}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")
