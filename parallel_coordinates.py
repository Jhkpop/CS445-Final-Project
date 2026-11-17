#!/usr/bin/env python3
"""
Interactive Parallel Coordinates Viewer
=======================================
Built with tkinter (GUI) + matplotlib (plotting). Requires numpy (a matplotlib
dependency), so no extra installs are needed.

Features
--------
* Load any CSV (numeric columns become axes, text columns can be used for colour)
  or use the built-in sample dataset.
* Show / hide axes, re-order them (Up / Down), invert individual axes.
* Colour lines by a categorical or numeric column, choose the colormap.
* Per-axis min-max normalisation or one shared scale for every axis.
* Straight or smooth (curved) lines, adjustable opacity and line width.
* A dotted line links the base of the first axis to the base of the second axis
  (can be toggled off).
* Brushing: left-click and drag on an axis to select a value range.
  Lines that pass through all brushes are highlighted, others are greyed out.
  Right-click an axis (or use "Clear brushes") to remove brushes.

Run:  python parallel_coordinates.py
"""

import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import colormaps
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

NONE_LABEL = "(none)"
CMAPS = ["viridis", "plasma", "inferno", "turbo", "coolwarm", "RdYlBu_r", "Spectral", "cividis"]
BRUSH_TOLERANCE = 0.12      # how close (in axis units) a click must be to an axis
N_TICKS = 5


# --------------------------------------------------------------------------- #
# Data handling
# --------------------------------------------------------------------------- #
class Dataset:
    """Holds numeric columns (float arrays) and categorical columns (string lists)."""

    def __init__(self, numeric, categorical, name="dataset"):
        self.numeric = numeric            # dict: name -> np.ndarray
        self.categorical = categorical    # dict: name -> list[str]
        self.name = name

    @property
    def n_rows(self):
        return len(next(iter(self.numeric.values())))

    @classmethod
    def from_csv(cls, path):
        with open(path, newline="", encoding="utf-8-sig") as f:
            sample = f.read(4096)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            except csv.Error:
                dialect = csv.excel
            rows = [r for r in csv.reader(f, dialect) if r]

        if len(rows) < 2:
            raise ValueError("The file needs a header row and at least one data row.")

        header = [h.strip() or f"col{i}" for i, h in enumerate(rows[0])]
        body = [r for r in rows[1:] if len(r) >= len(header)]
        raw = {h: [r[i].strip() for r in body] for i, h in enumerate(header)}

        numeric, categorical = {}, {}
        for h, values in raw.items():
            try:
                numeric[h] = np.array([float(v) if v != "" else np.nan for v in values])
            except ValueError:
                categorical[h] = values

        if len(numeric) < 2:
            raise ValueError("Need at least two numeric columns to draw parallel coordinates.")

        # Drop rows containing missing numeric values
        keep = np.all(np.column_stack(list(numeric.values())) == np.column_stack(list(numeric.values())), axis=1)
        if not keep.all():
            numeric = {k: v[keep] for k, v in numeric.items()}
            categorical = {k: [x for x, ok in zip(v, keep) if ok] for k, v in categorical.items()}

        return cls(numeric, categorical, name=path.replace("\\", "/").split("/")[-1])

    @classmethod
    def sample(cls):
        """Iris-like synthetic data (3 species x 50 samples)."""
        rng = np.random.default_rng(0)
        specs = {
            "setosa":     ([5.0, 3.4, 1.5, 0.25], [0.35, 0.38, 0.17, 0.10]),
            "versicolor": ([5.9, 2.8, 4.3, 1.30], [0.50, 0.31, 0.47, 0.20]),
            "virginica":  ([6.6, 3.0, 5.5, 2.00], [0.60, 0.32, 0.55, 0.27]),
        }
        blocks, species = [], []
        for sp, (mean, sd) in specs.items():
            blocks.append(np.abs(rng.normal(mean, sd, size=(50, 4))))
            species += [sp] * 50
        data = np.vstack(blocks)
        names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        numeric = {n: data[:, i] for i, n in enumerate(names)}
        return cls(numeric, {"species": species}, name="sample (iris-like)")


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
class ParallelCoordinatesApp:
    def __init__(self, root):
        self.root = root
        root.title("Parallel Coordinates Viewer")
        root.geometry("1250x720")

        self.ds = None
        self.order = []             # axis order (all numeric names)
        self.visible = set()        # names currently shown
        self.inverted = set()       # names with flipped direction
        self.brushes = {}           # name -> (lo, hi) in normalised display coords [0, 1]

        # drawing state used by mouse interaction
        self.ax = None
        self.drawn_cols = []
        self.drag = None            # (col_index, start_y, patch)

        # tk variables
        self.color_by = tk.StringVar(value=NONE_LABEL)
        self.cmap = tk.StringVar(value="viridis")
        self.normalize = tk.BooleanVar(value=True)
        self.smooth = tk.BooleanVar(value=False)
        self.base_link = tk.BooleanVar(value=True)
        self.alpha = tk.DoubleVar(value=0.6)
        self.linewidth = tk.DoubleVar(value=1.2)
        self.status = tk.StringVar(value="")

        self._build_ui()
        self.set_dataset(Dataset.sample())

    # ------------------------------------------------------------------ UI --
    def _build_ui(self):
        panel = ttk.Frame(self.root, padding=8)
        panel.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Button(panel, text="Load CSV…", command=self.load_csv).pack(fill=tk.X)
        ttk.Button(panel, text="Load sample data",
                   command=lambda: self.set_dataset(Dataset.sample())).pack(fill=tk.X, pady=(4, 8))

        # Axes list
        ttk.Label(panel, text="Axes  (double-click = show/hide)").pack(anchor=tk.W)
        lb_frame = ttk.Frame(panel)
        lb_frame.pack(fill=tk.X)
        self.listbox = tk.Listbox(lb_frame, height=10, exportselection=False, activestyle="none")
        sb = ttk.Scrollbar(lb_frame, command=self.listbox.yview)
        self.listbox.config(yscrollcommand=sb.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.bind("<Double-Button-1>", lambda e: self.toggle_visible())

        btns = ttk.Frame(panel)
        btns.pack(fill=tk.X, pady=4)
        for text, cmd in (("▲ Up", lambda: self.move(-1)), ("▼ Down", lambda: self.move(1)),
                          ("Show/Hide", self.toggle_visible), ("Invert", self.invert_axis)):
            ttk.Button(btns, text=text, width=9, command=cmd).pack(side=tk.LEFT, padx=1)

        ttk.Separator(panel).pack(fill=tk.X, pady=8)

        # Colour
        ttk.Label(panel, text="Colour by").pack(anchor=tk.W)
        self.color_box = ttk.Combobox(panel, textvariable=self.color_by, state="readonly")
        self.color_box.pack(fill=tk.X)
        self.color_box.bind("<<ComboboxSelected>>", lambda e: self.redraw())

        ttk.Label(panel, text="Colormap (numeric colouring)").pack(anchor=tk.W, pady=(6, 0))
        cmap_box = ttk.Combobox(panel, textvariable=self.cmap, values=CMAPS, state="readonly")
        cmap_box.pack(fill=tk.X)
        cmap_box.bind("<<ComboboxSelected>>", lambda e: self.redraw())

        ttk.Separator(panel).pack(fill=tk.X, pady=8)

        # Style
        ttk.Checkbutton(panel, text="Normalise each axis", variable=self.normalize,
                        command=self.structural_change).pack(anchor=tk.W)
        ttk.Checkbutton(panel, text="Smooth lines", variable=self.smooth,
                        command=self.redraw).pack(anchor=tk.W)
        ttk.Checkbutton(panel, text="Dotted link: axis 1 base → axis 2 base",
                        variable=self.base_link, command=self.redraw).pack(anchor=tk.W)

        ttk.Label(panel, text="Opacity").pack(anchor=tk.W, pady=(6, 0))
        ttk.Scale(panel, from_=0.02, to=1.0, variable=self.alpha,
                  command=lambda v: self.redraw()).pack(fill=tk.X)
        ttk.Label(panel, text="Line width").pack(anchor=tk.W, pady=(6, 0))
        ttk.Scale(panel, from_=0.3, to=4.0, variable=self.linewidth,
                  command=lambda v: self.redraw()).pack(fill=tk.X)

        ttk.Separator(panel).pack(fill=tk.X, pady=8)
        ttk.Button(panel, text="Clear brushes", command=self.clear_brushes).pack(fill=tk.X)
        ttk.Label(panel, text="Brush: drag on an axis.\nRight-click an axis to clear it.",
                  foreground="#555").pack(anchor=tk.W, pady=6)
        ttk.Label(panel, textvariable=self.status, wraplength=210, foreground="#222").pack(anchor=tk.W)

        # Plot area
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)

    # ---------------------------------------------------------- data/state --
    def load_csv(self):
        path = filedialog.askopenfilename(
            title="Open CSV", filetypes=[("CSV / text", "*.csv *.tsv *.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.set_dataset(Dataset.from_csv(path))
        except Exception as exc:
            messagebox.showerror("Could not load file", str(exc))

    def set_dataset(self, ds):
        self.ds = ds
        self.order = list(ds.numeric.keys())
        self.visible = set(self.order)
        self.inverted.clear()
        self.brushes.clear()

        options = [NONE_LABEL] + list(ds.categorical) + list(ds.numeric)
        self.color_box["values"] = options
        if ds.categorical:
            self.color_by.set(next(iter(ds.categorical)))
        else:
            self.color_by.set(self.order[0])

        self.refresh_listbox(select=0)
        self.redraw()

    def refresh_listbox(self, select=None):
        self.listbox.delete(0, tk.END)
        for name in self.order:
            box = "☑" if name in self.visible else "☐"
            flip = "  ↕" if name in self.inverted else ""
            self.listbox.insert(tk.END, f"{box} {name}{flip}")
        if select is not None and self.order:
            select = max(0, min(select, len(self.order) - 1))
            self.listbox.selection_set(select)
            self.listbox.see(select)

    def selected_index(self):
        sel = self.listbox.curselection()
        return sel[0] if sel else None

    def structural_change(self):
        """Any change that alters axis coordinates invalidates existing brushes."""
        self.brushes.clear()
        self.redraw()

    def toggle_visible(self):
        i = self.selected_index()
        if i is None:
            return
        name = self.order[i]
        self.visible.symmetric_difference_update({name})
        self.refresh_listbox(select=i)
        self.structural_change()

    def invert_axis(self):
        i = self.selected_index()
        if i is None:
            return
        name = self.order[i]
        self.inverted.symmetric_difference_update({name})
        self.refresh_listbox(select=i)
        self.structural_change()

    def move(self, step):
        i = self.selected_index()
        if i is None:
            return
        j = i + step
        if not 0 <= j < len(self.order):
            return
        self.order[i], self.order[j] = self.order[j], self.order[i]
        self.refresh_listbox(select=j)
        self.structural_change()

    def clear_brushes(self):
        self.brushes.clear()
        self.redraw()

    # ------------------------------------------------------------- drawing --
    def compute_layout(self, cols):
        """Return (Y normalised to [0,1] for display, lo, span) for the visible columns."""
        X = np.column_stack([self.ds.numeric[c] for c in cols])
        if self.normalize.get():
            lo, hi = X.min(axis=0), X.max(axis=0)
        else:
            lo = np.full(X.shape[1], X.min())
            hi = np.full(X.shape[1], X.max())
        span = np.where(hi - lo == 0, 1.0, hi - lo)
        Y = (X - lo) / span
        for j, c in enumerate(cols):
            if c in self.inverted:
                Y[:, j] = 1.0 - Y[:, j]
        return Y, lo, span

    def build_colors(self):
        """Return (rgba array, legend handles or None, ScalarMappable or None)."""
        n = self.ds.n_rows
        key = self.color_by.get()
        alpha = float(self.alpha.get())

        if key in self.ds.categorical:
            values = self.ds.categorical[key]
            cats = sorted(set(values))
            palette = colormaps["tab10" if len(cats) <= 10 else "tab20"]
            lookup = {c: palette(i % palette.N) for i, c in enumerate(cats)}
            rgba = np.array([to_rgba(lookup[v], alpha) for v in values])
            handles = [Line2D([0], [0], color=lookup[c], lw=3, label=c) for c in cats]
            return rgba, handles, None

        if key in self.ds.numeric:
            v = self.ds.numeric[key]
            norm = Normalize(vmin=v.min(), vmax=v.max() if v.max() > v.min() else v.min() + 1)
            cmap = colormaps[self.cmap.get()]
            rgba = cmap(norm(v))
            rgba[:, 3] = alpha
            return rgba, None, ScalarMappable(norm=norm, cmap=cmap)

        return np.tile(to_rgba("#1f77b4", alpha), (n, 1)), None, None

    def build_segments(self, Y):
        n, m = Y.shape
        if not self.smooth.get():
            xs = np.arange(m, dtype=float)
            return np.stack([np.broadcast_to(xs, (n, m)), Y], axis=2)
        t = np.linspace(0, 1, 14)
        ease = t * t * (3 - 2 * t)           # smoothstep: flat tangents at the axes
        xs = np.concatenate([j + t for j in range(m - 1)])
        ys = np.concatenate(
            [Y[:, j:j + 1] + (Y[:, j + 1:j + 2] - Y[:, j:j + 1]) * ease[None, :] for j in range(m - 1)],
            axis=1)
        return np.stack([np.broadcast_to(xs, ys.shape), ys], axis=2)

    def brush_mask(self, Y, cols):
        mask = np.ones(len(Y), dtype=bool)
        for j, c in enumerate(cols):
            if c in self.brushes:
                lo, hi = self.brushes[c]
                mask &= (Y[:, j] >= lo) & (Y[:, j] <= hi)
        return mask

    def redraw(self):
        if self.ds is None:
            return
        self.fig.clf()
        cols = [c for c in self.order if c in self.visible]
        self.drawn_cols = cols
        self.drag = None

        if len(cols) < 2:
            self.ax = None
            self.fig.text(0.5, 0.5, "Show at least two axes", ha="center", va="center",
                          fontsize=14, color="gray")
            self.status.set("")
            self.canvas.draw_idle()
            return

        Y, lo, span = self.compute_layout(cols)
        rgba, handles, mappable = self.build_colors()
        segments = self.build_segments(Y)
        mask = self.brush_mask(Y, cols)
        m = len(cols)

        # Figure layout (manual so legend / colourbar never collide with the plot)
        bottom = 0.14 if handles else 0.06
        width = 0.86 if mappable else 0.94
        ax = self.fig.add_axes([0.03, bottom, width, 0.80 - (bottom - 0.06)])
        self.ax = ax

        lw = float(self.linewidth.get())
        if (~mask).any():
            ax.add_collection(LineCollection(segments[~mask], colors=(0.7, 0.7, 0.7, 0.15),
                                             linewidths=lw * 0.8, zorder=1))
        sel_colors = rgba[mask]
        if self.brushes:
            sel_colors = sel_colors.copy()
            sel_colors[:, 3] = np.maximum(sel_colors[:, 3], 0.8)
        ax.add_collection(LineCollection(segments[mask], colors=sel_colors,
                                         linewidths=lw * (1.2 if self.brushes else 1.0), zorder=2))

        # Axes, labels and ticks
        for j, c in enumerate(cols):
            ax.axvline(j, color="black", lw=1.6, zorder=4)
            label = c + (" ↕" if c in self.inverted else "")
            ax.text(j, 1.07, label, ha="center", va="bottom", fontsize=10, fontweight="bold")
            for t in np.linspace(0, 1, N_TICKS):
                frac = 1 - t if c in self.inverted else t
                value = lo[j] + frac * span[j]
                ax.plot([j - 0.025, j + 0.025], [t, t], color="black", lw=1, zorder=4)
                ax.text(j + 0.04, t, f"{value:.3g}", ha="left", va="center", fontsize=7,
                        color="#333", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))

        # Dotted line connecting the base of the first axis to the base of the second
        if self.base_link.get():
            ax.plot([0, 1], [0, 0], color="black", linestyle=":", linewidth=2.0,
                    marker="o", markersize=4, zorder=3)

        # Brush rectangles
        for j, c in enumerate(cols):
            if c in self.brushes:
                b_lo, b_hi = self.brushes[c]
                ax.add_patch(Rectangle((j - 0.07, b_lo), 0.14, b_hi - b_lo, fc="#ff7f0e",
                                       ec="#d35400", alpha=0.35, lw=1.2, zorder=6))

        ax.set_xlim(-0.35, m - 1 + 0.55)
        ax.set_ylim(-0.04, 1.06)
        ax.axis("off")

        if handles:
            self.fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6),
                            frameon=False, title=self.color_by.get())
        if mappable:
            cax = self.fig.add_axes([0.94, bottom, 0.015, 0.74])
            self.fig.colorbar(mappable, cax=cax, label=self.color_by.get())

        self.status.set(f"{self.ds.name}\n{int(mask.sum())} of {len(mask)} lines selected"
                        + ("" if self.brushes else "  (no brushes)"))
        self.canvas.draw_idle()

    # --------------------------------------------------------- interaction --
    def _axis_hit(self, event):
        if self.ax is None or event.inaxes is not self.ax or event.xdata is None:
            return None
        j = int(round(event.xdata))
        if 0 <= j < len(self.drawn_cols) and abs(event.xdata - j) <= BRUSH_TOLERANCE:
            return j
        return None

    def on_press(self, event):
        if self.toolbar.mode:               # zoom/pan tool active -> don't brush
            return
        j = self._axis_hit(event)
        if j is None:
            return
        if event.button == 3:               # right click clears that axis' brush
            self.brushes.pop(self.drawn_cols[j], None)
            self.redraw()
            return
        if event.button == 1:
            y = float(np.clip(event.ydata, 0, 1))
            patch = Rectangle((j - 0.07, y), 0.14, 0, fc="#ff7f0e", ec="#d35400",
                              alpha=0.35, lw=1.2, zorder=6)
            self.ax.add_patch(patch)
            self.drag = (j, y, patch)

    def on_motion(self, event):
        if self.drag is None or event.ydata is None:
            return
        j, y0, patch = self.drag
        y1 = float(np.clip(event.ydata, 0, 1))
        patch.set_y(min(y0, y1))
        patch.set_height(abs(y1 - y0))
        self.canvas.draw_idle()

    def on_release(self, event):
        if self.drag is None:
            return
        j, y0, _ = self.drag
        y1 = y0 if event.ydata is None else float(np.clip(event.ydata, 0, 1))
        name = self.drawn_cols[j]
        lo, hi = sorted((y0, y1))
        if hi - lo < 0.01:                  # a simple click removes the brush
            self.brushes.pop(name, None)
        else:
            self.brushes[name] = (lo, hi)
        self.drag = None
        self.redraw()


def main():
    root = tk.Tk()
    ParallelCoordinatesApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
