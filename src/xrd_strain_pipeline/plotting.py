"""Plotting helpers for manual peak-list inspection."""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.image as mpimg
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
import numpy as np

from .window_utils import normalize_detector_window


def load_peaklist(path: str | Path) -> np.ndarray:
    """Load a peak list with columns z_frame, y, x, intensity."""

    path = Path(path)
    try:
        peaklist = np.loadtxt(path, delimiter=",")
    except ValueError:
        peaklist = np.loadtxt(path)
    if peaklist.ndim == 1:
        peaklist = peaklist.reshape(1, -1)
    if peaklist.shape[1] != 4:
        raise ValueError("Peak list must have 4 columns: z_frame, y, x, intensity")
    return peaklist


def plot_peaklist(
    peaklist_path: str | Path,
    overlay_png: str | Path | None = None,
    sort_by_frame: bool = True,
    label_peaks: bool = True,
    point_size: float = 50,
    label_size: float = 24,
    figsize: tuple[float, float] = (12, 10),
    save_path: str | Path | None = None,
):
    """Plot manual peak positions, optionally over a detector PNG."""

    peaklist = load_peaklist(peaklist_path)
    if sort_by_frame:
        peaklist = peaklist[np.argsort(peaklist[:, 0])]

    z_frame = peaklist[:, 0]
    pixel_y = peaklist[:, 1]
    pixel_x = peaklist[:, 2]

    fig, ax = plt.subplots(figsize=figsize)
    if overlay_png is not None:
        image = mpimg.imread(overlay_png)
        ax.imshow(image, origin="upper")

    ax.scatter(pixel_x, pixel_y, s=point_size, color="blue", label="Peaks")

    if label_peaks:
        for i, (x, y, _z) in enumerate(zip(pixel_x, pixel_y, z_frame)):
            ax.text(x + 5, y + 5, str(i), color="red", fontsize=label_size)

    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    ax.set_title("Peak List Sorted by Increasing Frame Number")
    ax.grid(True)
    if overlay_png is None:
        ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax, peaklist


def plot_bragg_predictions(
    predictions,
    overlay_png: str | Path | None = None,
    detector_size: tuple[int, int] = (3450, 3450),
    show_labels: bool = True,
    point_size: float = 18,
    label_size: float = 5,
    figsize: tuple[float, float] = (10, 10),
    save_path: str | Path | None = None,
):
    """Plot predicted Bragg positions on an optional detector PNG."""

    fig, ax = plt.subplots(figsize=figsize)
    if overlay_png is not None:
        image = mpimg.imread(overlay_png)
        ax.imshow(image, cmap="gray", origin="upper")
    else:
        ax.imshow(np.zeros(detector_size), cmap="gray", origin="upper")

    if predictions:
        x = [item["x"] for item in predictions]
        y = [item["y"] for item in predictions]
        ax.scatter(
            x,
            y,
            s=point_size,
            facecolors="none",
            edgecolors="red",
            marker="o",
            linewidths=0.8,
            label="Predicted Bragg",
            zorder=3,
        )
        if show_labels:
            for item in predictions:
                label = f"({item['h']},{item['k']},{item['l']})"
                ax.text(
                    item["x"] + 8,
                    item["y"] - 8,
                    label,
                    color="red",
                    fontsize=label_size,
                    weight="bold",
                    zorder=4,
                )

    ax.set_xlim(0, detector_size[0])
    ax.set_ylim(detector_size[1], 0)
    ax.set_xlabel("Detector X (pixels)")
    ax.set_ylabel("Detector Y (pixels)")
    ax.set_title("Predicted Bragg Peak Positions")
    ax.grid(False)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_cdw_predictions(
    bragg_predictions,
    cdw_predictions,
    overlay_png: str | Path | None = None,
    detector_size: tuple[int, int] = (3450, 3450),
    show_bragg_labels: bool = True,
    show_cdw_labels: bool = False,
    bragg_size: float = 14,
    cdw_size: float = 14,
    label_size: float = 5,
    figsize: tuple[float, float] = (10, 10),
    save_path: str | Path | None = None,
):
    """Plot predicted Bragg and CDW satellite positions on a detector PNG."""

    fig, ax = plt.subplots(figsize=figsize)
    if overlay_png is not None:
        image = mpimg.imread(overlay_png)
        ax.imshow(image, cmap="gray", origin="upper")
    else:
        ax.imshow(np.zeros(detector_size), cmap="gray", origin="upper")

    if bragg_predictions:
        bragg_x = [item["x"] for item in bragg_predictions]
        bragg_y = [item["y"] for item in bragg_predictions]
        bragg = ax.scatter(
            bragg_x,
            bragg_y,
            s=bragg_size,
            facecolors="none",
            edgecolors="red",
            marker="o",
            linewidths=0.7,
            label="Predicted Bragg",
            zorder=3,
        )
        bragg.set_path_effects(
            [path_effects.Stroke(linewidth=1.0, foreground="white", alpha=0.35), path_effects.Normal()]
        )
        if show_bragg_labels:
            for item in bragg_predictions:
                label = f"({item['h']:.2g},{item['k']:.2g},{item['l']:.2g})"
                ax.text(
                    item["x"] + 8,
                    item["y"] - 8,
                    label,
                    color="red",
                    fontsize=label_size,
                    weight="bold",
                    zorder=4,
                )

    if cdw_predictions:
        cdw_x = [item["x"] for item in cdw_predictions]
        cdw_y = [item["y"] for item in cdw_predictions]
        cdw = ax.scatter(
            cdw_x,
            cdw_y,
            s=cdw_size,
            facecolors="none",
            edgecolors="cyan",
            marker="^",
            linewidths=0.7,
            label="Predicted CDW",
            zorder=3,
        )
        cdw.set_path_effects(
            [path_effects.Stroke(linewidth=1.0, foreground="black", alpha=0.35), path_effects.Normal()]
        )
        if show_cdw_labels:
            for item in cdw_predictions:
                label = f"({item['h']:.2f},{item['k']:.2f},{item['l']:.2f})"
                ax.text(
                    item["x"] + 8,
                    item["y"] - 8,
                    label,
                    color="deepskyblue",
                    fontsize=label_size,
                    weight="bold",
                    zorder=4,
                )

    ax.set_xlim(0, detector_size[0])
    ax.set_ylim(detector_size[1], 0)
    ax.set_xlabel("Detector X (pixels)")
    ax.set_ylabel("Detector Y (pixels)")
    ax.set_title("Predicted CDW and Bragg Peak Positions")
    ax.grid(False)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def resolve_overlay_png(
    overlay_root: str | Path,
    scan_id: str,
    overlay_file: str | Path | None = None,
) -> Path:
    """Resolve an explicit or scan-named detector overlay PNG."""

    overlay_root = Path(overlay_root)
    if overlay_file is not None:
        path = overlay_root / overlay_file
    else:
        path = overlay_root / f"{scan_id}.png"
        if not path.exists():
            available = sorted(overlay_root.glob("*.png"))
            if len(available) == 1:
                path = available[0]
                print(f"Using the only available overlay: {path.name}")
    if not path.exists():
        raise FileNotFoundError(f"Missing detector overlay PNG: {path}")
    return path


def select_detector_window_interactive(
    overlay_png: str | Path,
    initial_bounds: tuple[int, int, int, int] | None = None,
    center: tuple[float, float] | None = None,
    backend: str | None = "TkAgg",
) -> tuple[int, int, int, int]:
    """Open a GUI detector image and return a manually drawn X/Y rectangle."""

    overlay_png = Path(overlay_png)
    if not overlay_png.exists():
        raise FileNotFoundError(f"Missing detector overlay PNG: {overlay_png}")
    previous_backend = plt.get_backend()
    if backend is not None:
        try:
            plt.switch_backend(backend)
        except (ImportError, RuntimeError) as error:
            raise RuntimeError(
                f"Could not start Matplotlib backend {backend!r}. "
                "Enable a desktop Matplotlib backend or pass backend=None."
            ) from error

    image = mpimg.imread(overlay_png)
    image_height, image_width = image.shape[:2]
    selected = {"bounds": None, "cancelled": False}

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(image, cmap="gray", origin="upper")
    ax.set_xlim(0, image_width)
    ax.set_ylim(image_height, 0)
    ax.set_xlabel("Detector X (pixels)")
    ax.set_ylabel("Detector Y (pixels)")
    ax.set_title("Left-drag select | Wheel zoom | Right-drag pan | Z selection | R reset | Enter accept")
    if center is not None:
        ax.scatter(center[0], center[1], marker="+", s=100, color="red", linewidths=1.2, zorder=4)

    def save_bounds(x0, x1, y0, y1):
        bounds = normalize_detector_window(
            int(np.floor(x0)),
            int(np.ceil(x1)),
            int(np.floor(y0)),
            int(np.ceil(y1)),
        )
        x0n, x1n, y0n, y1n = bounds
        if x0n < 0 or y0n < 0 or x1n > image_width or y1n > image_height:
            raise ValueError("Selected rectangle extends beyond the detector image")
        selected["bounds"] = bounds
        ax.set_title(f"Selected x=[{x0n}, {x1n}), y=[{y0n}, {y1n}); Enter accepts")
        fig.canvas.draw_idle()

    def on_select(click, release):
        save_bounds(click.xdata, release.xdata, click.ydata, release.ydata)

    selector = RectangleSelector(
        ax,
        on_select,
        useblit=True,
        button=[1],
        minspanx=1,
        minspany=1,
        spancoords="pixels",
        interactive=True,
    )

    if initial_bounds is not None:
        # Matplotlib 3.3 requires a cached renderer before updating selector artists.
        fig.canvas.draw()
        xstart, xend, ystart, yend = normalize_detector_window(*initial_bounds)
        selector.extents = (xstart, xend, ystart, yend)
        selected["bounds"] = (xstart, xend, ystart, yend)

    pan_state = {"active": False}

    def on_scroll(event):
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        scale = 1 / 1.5 if event.button == "up" else 1.5
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        width = (x1 - x0) * scale
        height = (y1 - y0) * scale
        x_fraction = (event.xdata - x0) / (x1 - x0)
        y_fraction = (event.ydata - y0) / (y1 - y0)
        ax.set_xlim(event.xdata - width * x_fraction, event.xdata + width * (1 - x_fraction))
        ax.set_ylim(event.ydata - height * y_fraction, event.ydata + height * (1 - y_fraction))
        fig.canvas.draw_idle()

    def on_press(event):
        if event.button != 3 or event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        pan_state.update(
            active=True,
            x=event.xdata,
            y=event.ydata,
            xlim=ax.get_xlim(),
            ylim=ax.get_ylim(),
        )

    def on_motion(event):
        if not pan_state["active"] or event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        dx = event.xdata - pan_state["x"]
        dy = event.ydata - pan_state["y"]
        ax.set_xlim(pan_state["xlim"][0] - dx, pan_state["xlim"][1] - dx)
        ax.set_ylim(pan_state["ylim"][0] - dy, pan_state["ylim"][1] - dy)
        fig.canvas.draw_idle()

    def on_release(event):
        if event.button == 3:
            pan_state["active"] = False

    def on_key(event):
        if event.key == "enter":
            if selected["bounds"] is None:
                x0, x1, y0, y1 = selector.extents
                save_bounds(x0, x1, y0, y1)
            plt.close(fig)
        elif event.key == "escape":
            selected["cancelled"] = True
            plt.close(fig)
        elif event.key in {"r", "R"}:
            ax.set_xlim(0, image_width)
            ax.set_ylim(image_height, 0)
            fig.canvas.draw_idle()
        elif event.key in {"z", "Z"} and selected["bounds"] is not None:
            xstart, xend, ystart, yend = selected["bounds"]
            margin = max(10, int(max(xend - xstart, yend - ystart) * 0.5))
            ax.set_xlim(max(0, xstart - margin), min(image_width, xend + margin))
            ax.set_ylim(min(image_height, yend + margin), max(0, ystart - margin))
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    plt.show(block=True)

    if backend is not None and plt.get_backend().lower() != previous_backend.lower():
        plt.switch_backend(previous_backend)

    if selected["cancelled"]:
        raise RuntimeError("Interactive detector-window selection was cancelled")
    if selected["bounds"] is None:
        raise RuntimeError("No detector window was selected")
    return selected["bounds"]


def plot_detector_window(
    overlay_png: str | Path,
    xstart: int,
    xend: int,
    ystart: int,
    yend: int,
    center: tuple[float, float] | None = None,
    crop_margin: int = 40,
    save_path: str | Path | None = None,
):
    """Show a detector window on full and zoomed max-projection views."""

    overlay_png = Path(overlay_png)
    if not overlay_png.exists():
        raise FileNotFoundError(f"Missing detector overlay PNG: {overlay_png}")
    xstart, xend, ystart, yend = normalize_detector_window(xstart, xend, ystart, yend)

    image = mpimg.imread(overlay_png)
    image_height, image_width = image.shape[:2]
    if xstart < 0 or ystart < 0 or xend > image_width or yend > image_height:
        raise ValueError(
            f"Window x=[{xstart}, {xend}), y=[{ystart}, {yend}) exceeds "
            f"image size {image_width}x{image_height}"
        )
    if center is not None and not (xstart <= center[0] < xend and ystart <= center[1] < yend):
        warnings.warn(
            f"Peak center {center} is outside the selected detector window; plotting it only on the full view.",
            UserWarning,
            stacklevel=2,
        )

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax in axes:
        ax.imshow(image, cmap="gray", origin="upper")
        ax.add_patch(
            Rectangle(
                (xstart, ystart),
                xend - xstart,
                yend - ystart,
                fill=False,
                edgecolor="cyan",
                linewidth=1.5,
                zorder=3,
            )
        )
        if center is not None:
            ax.scatter(center[0], center[1], marker="+", s=90, color="red", linewidths=1.2, zorder=4)
        ax.set_xlabel("Detector X (pixels)")
        ax.set_ylabel("Detector Y (pixels)")

    axes[0].set_xlim(0, image_width)
    axes[0].set_ylim(image_height, 0)
    axes[0].set_title("Full Detector")

    axes[1].set_xlim(max(0, xstart - crop_margin), min(image_width, xend + crop_margin))
    axes[1].set_ylim(min(image_height, yend + crop_margin), max(0, ystart - crop_margin))
    axes[1].set_title(
        f"Selected Window: x=[{xstart}, {xend}), y=[{ystart}, {yend})"
    )

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes


def plot_lab_momentum_maps(
    nxprocess_lab,
    transform_data,
    hrange,
    krange,
    lrange,
    hval: float | None = None,
    kval: float | None = None,
    lval: float | None = None,
    save_dir: str | Path | None = None,
    filename_prefix: str = "momentum",
):
    """Call the lab HK/KL/HL map functions and report selected-plane occupancy."""

    max_index = np.unravel_index(np.nanargmax(transform_data), transform_data.shape)
    hval = float(hrange[max_index[0]]) if hval is None else float(hval)
    kval = float(krange[max_index[1]]) if kval is None else float(kval)
    lval = float(lrange[max_index[2]]) if lval is None else float(lval)

    def plane_index(axis, value):
        return min(int(np.searchsorted(axis, value, side="left")), len(axis) - 1)

    hindex = plane_index(hrange, hval)
    kindex = plane_index(krange, kval)
    lindex = plane_index(lrange, lval)
    planes = {
        "HK": transform_data[:, :, lindex],
        "KL": transform_data[hindex, :, :],
        "HL": transform_data[:, kindex, :],
    }
    coverage = {
        name: {
            "nonzero_fraction": float(np.count_nonzero(plane) / plane.size),
            "zero_fraction": float(1.0 - np.count_nonzero(plane) / plane.size),
        }
        for name, plane in planes.items()
    }

    figures = {}
    nxprocess_lab.HKmap(Hrange=hrange, Krange=krange, Lrange=lrange, Lval=lval, transform_data=transform_data)
    figures["HK"] = plt.gcf()
    nxprocess_lab.KLmap(Hrange=hrange, Krange=krange, Lrange=lrange, Hval=hval, transform_data=transform_data)
    figures["KL"] = plt.gcf()
    nxprocess_lab.HLmap(Hrange=hrange, Krange=krange, Lrange=lrange, Kval=kval, transform_data=transform_data)
    figures["HL"] = plt.gcf()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            fig.savefig(save_dir / f"{filename_prefix}_{name}.png", dpi=300, bbox_inches="tight")

    print(f"Map planes: H={hval:.6f}, K={kval:.6f}, L={lval:.6f}")
    for name in ("HK", "KL", "HL"):
        print(f"{name} plane zero bins: {coverage[name]['zero_fraction']:.1%}")
    return {
        "figures": figures,
        "coverage": coverage,
        "hval": hval,
        "kval": kval,
        "lval": lval,
    }


def plot_momentum_line_scans(
    line_scans,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (15, 4),
    title: str = "Momentum Line Scans",
    color: str | None = None,
    background_y: float | dict[str, float] | None = None,
    background_label: str = "Background",
    integration_bounds: dict[str, float] | None = None,
):
    """Plot H/K/L line scans returned by momentum_line_scans."""

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    labels = {
        "H": "H (r.l.u.)",
        "K": "K (r.l.u.)",
        "L": "L (r.l.u.)",
    }
    for ax, axis_name in zip(axes, ("H", "K", "L")):
        scan = line_scans[axis_name]
        axis_background = (
            background_y.get(axis_name)
            if isinstance(background_y, dict)
            else background_y
        )
        ax.plot(scan["axis"], scan["intensity"], "o:", color=color)
        if integration_bounds is not None:
            key = axis_name.lower()
            lower = integration_bounds.get(f"{key}min")
            upper = integration_bounds.get(f"{key}max")
            if lower is not None and upper is not None:
                ax.axvspan(
                    lower,
                    upper,
                    color="tab:green",
                    alpha=0.15,
                    label="Integration ROI",
                )
        if axis_background is not None:
            ax.axhline(
                axis_background,
                color="tab:red",
                linestyle="--",
                linewidth=1.2,
                label=background_label,
            )
        if integration_bounds is not None or axis_background is not None:
            ax.legend(fontsize=8)
        ax.set_xlabel(labels[axis_name])
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title(f"{axis_name} Scan")
        ax.tick_params(labelsize=10)
    fig.suptitle(
        f"{title} at "
        f"H={line_scans['center_hkl'][0]:.6f}, "
        f"K={line_scans['center_hkl'][1]:.6f}, "
        f"L={line_scans['center_hkl'][2]:.6f}"
    )
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes
