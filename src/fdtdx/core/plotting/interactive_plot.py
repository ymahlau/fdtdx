import jax
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RadioButtons, Slider


def plot_interactive_3d_slices(arr: jax.Array | np.ndarray):
    """
    Creates an interactive matplotlib window to slice through a 3D array.
    Works with both NumPy and JAX arrays.
    """
    # Convert JAX/other array types to numpy for matplotlib compatibility
    arr = np.asarray(arr)

    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D array, got {arr.ndim}D.")

    # Initial state variables
    current_axis = 0
    max_idx = arr.shape[current_axis] - 1
    init_idx = max_idx // 2

    # Set up the figure and main image axis
    fig, ax = plt.subplots(figsize=(8, 8))
    # Shrink the main plot to make room for the UI elements
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Plot the initial slice
    # vmin/vmax lock the color scale so it doesn't flicker when changing slices
    vmin, vmax = np.min(arr), np.max(arr)
    im = ax.imshow(arr[init_idx, :, :], cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(f"Axis {current_axis}, Slice {init_idx}/{max_idx}")

    # --- UI: Slider ---
    ax_slider = plt.axes((0.25, 0.1, 0.65, 0.03))
    slider = Slider(ax=ax_slider, label="Slice Index", valmin=0, valmax=max_idx, valinit=init_idx, valstep=1)

    # --- UI: Radio Buttons ---
    ax_radio = plt.axes((0.02, 0.4, 0.15, 0.15))
    radio = RadioButtons(ax=ax_radio, labels=("Axis 0", "Axis 1", "Axis 2"), active=0)

    # --- Callback Functions ---
    def update_slice(val=None):
        """Updates the image when the slider is moved."""
        axis_str = radio.value_selected
        axis_idx = int(axis_str.split()[-1])  # Parse 'Axis 0' -> 0
        slice_idx = int(slider.val)

        # Extract the correct 2D slice based on selected axis
        if axis_idx == 0:
            slice_data = arr[slice_idx, :, :]
        elif axis_idx == 1:
            slice_data = arr[:, slice_idx, :]
        else:
            slice_data = arr[:, :, slice_idx]

        im.set_data(slice_data)
        ax.set_title(f"Axis {axis_idx}, Slice {slice_idx}/{arr.shape[axis_idx] - 1}")
        fig.canvas.draw_idle()

    def update_axis(label):
        """Updates the slider limits when a new axis is selected."""
        axis_idx = int(label.split()[-1])
        new_max = arr.shape[axis_idx] - 1

        # Update slider limits dynamically
        slider.valmax = new_max
        slider.ax.set_xlim(slider.valmin, new_max)

        # Prevent index out of bounds if switching to a smaller axis
        if slider.val > new_max:
            slider.set_val(new_max // 2)

        update_slice()

    # Connect the UI elements to their callbacks
    slider.on_changed(update_slice)
    radio.on_clicked(update_axis)

    # IMPORTANT: return the widgets, or Python's garbage collector
    # will delete them and the buttons/slider will become unresponsive.
    return fig, ax, slider, radio


def plot_interactive_3d_cutoff(arr) -> None:
    """
    Interactive 3D cut-away visualisation using PyVista.

    Only the *shell* of the visible half-space is rendered — that is, every
    voxel that has at least one face exposed to empty space.  This covers
    both the outer surface of the block and the cross-section at the cutoff
    plane, while completely skipping fully-interior voxels that would never
    be visible anyway.

    Performance notes
    -----------------
    * PyVista/VTK renders the geometry as a single GPU mesh, so the scene
      stays interactive even for arrays up to ~200³.
    * Shell extraction is a pure NumPy operation (one pad + six shifts)
      and runs in microseconds.
    * Mesh updates reuse the same actor via ``name=`` so there is no
      actor accumulation between slider moves.

    Parameters
    ----------
    arr : array-like, shape (Nx, Ny, Nz)
        3-D array.  Accepts NumPy or JAX arrays (converted automatically).
        Values are colour-mapped with viridis.
    """
    try:
        import pyvista as pv
    except ImportError:
        raise Exception("Need to install pyvista to run interactive cutoff visualization")

    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3-D array, got shape {arr.shape}.")

    Nx, Ny, Nz = arr.shape

    # ── Static base grid ─────────────────────────────────────────────────────
    # ImageData dimensions are *corner points*; the number of cells per axis
    # is therefore (dim - 1).  VTK numbers cells x-fastest → Fortran ravel.
    grid = pv.ImageData(dimensions=(Nx + 1, Ny + 1, Nz + 1))
    grid.cell_data["values"] = arr.ravel(order="F")

    # ── Shared mutable state ──────────────────────────────────────────────────
    state: dict = {
        "axis": 0,
        "side": "Left",
        "cutoff": arr.shape[0] // 2,
    }

    # ── Shell extraction ──────────────────────────────────────────────────────
    def build_shell_mesh() -> pv.UnstructuredGrid:
        axis = state["axis"]
        side = state["side"]
        cutoff = state["cutoff"]

        # Build the half-space boolean mask
        mask = np.zeros((Nx, Ny, Nz), dtype=bool)
        sl = [slice(None), slice(None), slice(None)]
        sl[axis] = slice(None, cutoff + 1) if side == "Left" else slice(cutoff + 1, None)
        mask[tuple(sl)] = True

        # A voxel belongs to the shell iff it is inside the mask AND at least
        # one of its six face-neighbours is outside the mask.
        p = np.pad(mask, 1, constant_values=False)
        exposed = (
            ~p[:-2, 1:-1, 1:-1]
            | ~p[2:, 1:-1, 1:-1]  # ± x
            | ~p[1:-1, :-2, 1:-1]
            | ~p[1:-1, 2:, 1:-1]  # ± y
            | ~p[1:-1, 1:-1, :-2]
            | ~p[1:-1, 1:-1, 2:]  # ± z
        )
        shell = mask & exposed

        cell_ids = np.flatnonzero(shell.ravel(order="F"))
        if cell_ids.size == 0:
            return pv.UnstructuredGrid()  # nothing to draw (edge case)
        return grid.extract_cells(cell_ids)

    # ── Plotter ───────────────────────────────────────────────────────────────
    pl = pv.Plotter()
    pl.add_axes()  # ty:ignore[missing-argument]
    pl.set_background("#2b2b2b")  # ty:ignore[invalid-argument-type]

    MESH_NAME = "shell"

    def redraw(_=None) -> None:
        """Rebuild and replace the shell mesh in-place (no actor leakage)."""
        mesh = build_shell_mesh()

        if mesh.n_cells == 0:
            pl.remove_actor(MESH_NAME)  # ty:ignore[invalid-argument-type]
            return

        pl.add_mesh(
            mesh,
            name=MESH_NAME,  # same name → replaces the existing actor
            scalars="values",
            cmap="viridis",
            show_edges=True,
            edge_color="black",
            line_width=0.3,
            scalar_bar_args={"title": "Value", "vertical": True},
        )

    redraw()

    # ── Widgets ───────────────────────────────────────────────────────────────

    # Cutoff index slider  (horizontal, bottom)
    def on_cutoff(value: float) -> None:
        state["cutoff"] = round(value)
        redraw()

    cutoff_slider = pl.add_slider_widget(
        callback=on_cutoff,
        rng=[0, arr.shape[state["axis"]] - 1],
        value=state["cutoff"],
        title="Cutoff index",
        pointa=(0.10, 0.06),
        pointb=(0.80, 0.06),  # Slightly shortened to avoid overlapping the vertical colorbar
        style="modern",
        interaction_event="always",
    )

    # Lists to keep track of the checkbox widgets so we can toggle them programmatically
    axis_widgets = []
    side_widgets = []

    def update_axis(idx: int):
        state["axis"] = idx
        # Ensure radio-button visual behavior
        for i, w in enumerate(axis_widgets):
            w.GetRepresentation().SetState(1 if i == idx else 0)

        new_max = arr.shape[state["axis"]] - 1
        state["cutoff"] = min(state["cutoff"], new_max)
        rep = cutoff_slider.GetRepresentation()
        rep.SetMaximumValue(float(new_max))
        rep.SetValue(float(state["cutoff"]))
        redraw()

    def make_axis_callback(idx: int):
        def callback(state_bool: bool):
            if not state_bool and state["axis"] == idx:
                # Prevent the user from unchecking the currently active button
                axis_widgets[idx].GetRepresentation().SetState(1)
            elif state_bool:
                update_axis(idx)

        return callback

    def update_side(idx: int, side_str: str):
        state["side"] = side_str
        # Ensure radio-button visual behavior
        for i, w in enumerate(side_widgets):
            w.GetRepresentation().SetState(1 if i == idx else 0)
        redraw()

    def make_side_callback(idx: int, side_str: str):
        def callback(state_bool: bool):
            if not state_bool and state["side"] == side_str:
                # Prevent the user from unchecking the currently active button
                side_widgets[idx].GetRepresentation().SetState(1)
            elif state_bool:
                update_side(idx, side_str)

        return callback

    # --- UI Placement Parameters (Pixels from bottom-left) ---
    Y_HEADER = 730
    Y_LABEL = 695
    Y_BTN = 660

    # Render Axis Radio Buttons (0, 1, 2)
    pl.add_text("Axis", position=(50, Y_HEADER), font_size=12, color="white")
    for i, label in enumerate(["0", "1", "2"]):
        x_pos = 50 + i * 60
        pl.add_text(label, position=(x_pos + 8, Y_LABEL), font_size=10, color="white")
        btn = pl.add_checkbox_button_widget(
            make_axis_callback(i),
            value=(state["axis"] == i),
            position=(x_pos, Y_BTN),
            size=30,
        )
        axis_widgets.append(btn)

    # Render Side Radio Buttons (Left, Right)
    pl.add_text("Side", position=(700, Y_HEADER), font_size=12, color="white")
    for i, label in enumerate(["Left", "Right"]):
        x_pos = 700 + i * 80
        pl.add_text(label, position=(x_pos + 5, Y_LABEL), font_size=10, color="white")
        btn = pl.add_checkbox_button_widget(
            make_side_callback(i, label),
            value=(state["side"] == label),
            position=(x_pos, Y_BTN),
            size=30,
        )
        side_widgets.append(btn)

    pl.show(title="3D Cutoff Viewer")
