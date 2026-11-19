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
* Adjustable opacity and line width.
* Axes are taken in consecutive pairs: (1,2), (3,4), (5,6), ... For each pair:
  - "Fan to origin": every value on the left axis is joined by a dotted line to
    the base (origin) of the right axis.
  - "Reverse fan": every value on the right axis is joined by a dotted line to
    the base (origin) of the left axis.
  - "Intersection points": each row's two fan lines cross at one point, which
    is marked with a dot. Everything follows the colouring and brushing.
  In the two paired layouts, if the number of visible axes is odd, the last axis
  (and its values) is duplicated so that every axis has a partner; the copy is
  labelled "<domain name> dup". Plain parallel coordinates never duplicates.
* Three layout buttons (the active one is greyed out):
  - "Parallel coordinates": ordinary, uncombined axes. No fans, no dots.
  - "Represented paired parallel coordinates": axes (2,3), (4,5), (6,7), ... share
    one vertical axis (first axis' ticks on the left, second axis' ticks on the
    right) so the fan pairs (1,2), (3,4), (5,6) ... chain together. The original
    lines, the fans and the intersection dots are drawn. Brushing works on either
    half of a shared axis.
  - "Paired parallel coordinates": the same layout, but the original lines and the
    fans are hidden; only the intersection dots and a line per row joining its
    dots are drawn: pair (1,2) -> pair (3,4) -> pair (5,6) -> ...

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
MODE_PC = "pc"          # parallel coordinates (uncombined axes)
MODE_RPPC = "rppc"      # represented paired parallel coordinates (combined axes)
MODE_PPC = "ppc"        # paired parallel coordinates (combined axes + dot-to-dot lines)
MODE_LABELS = [
    (MODE_PC, "Parallel coordinates"),
    (MODE_RPPC, "Represented paired parallel coordinates"),
    (MODE_PPC, "Paired parallel coordinates"),
]
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
        self.mode = MODE_PC         # layout mode, see MODE_* constants
        self.col_pos = []           # x position of each drawn column
        self.col_side = []          # -1 / +1 = left / right half of a combined axis, 0 = alone

        # tk variables
        self.color_by = tk.StringVar(value=NONE_LABEL)
        self.cmap = tk.StringVar(value="viridis")
        self.normalize = tk.BooleanVar(value=True)
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

        self.mode_btns = {}
        for mode, text in MODE_LABELS:
            btn = ttk.Button(panel, text=text, command=lambda m=mode: self.set_mode(m))
            btn.pack(fill=tk.X, pady=1)
            self.mode_btns[mode] = btn
        self._update_mode_buttons()

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
        self.mode = MODE_PC
        self._update_mode_buttons()

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

    @property
    def combined(self):
        return self.mode != MODE_PC

    def set_mode(self, mode):
        self.mode = mode
        self._update_mode_buttons()
        self.redraw()

    def _update_mode_buttons(self):
        for mode, btn in self.mode_btns.items():
            btn.config(state=tk.DISABLED if mode == self.mode else tk.NORMAL)

    def layout_positions(self, m):
        """x position and side (-1 / 0 / +1) for each of m visible columns.

        Uncombined: column k sits at x = k.  Combined: columns (1,2), (3,4), ... (0-based)
        share one x, i.e. axes 2+3, 4+5, ... in 1-based numbering.
        """
        if not self.combined:
            return list(range(m)), [0] * m
        pos, side = [0], [0]
        for k in range(1, m):
            pos.append((k + 1) // 2)
            if k % 2 == 1:
                side.append(-1 if k + 1 < m else 0)    # first of a combined pair (or a lone last axis)
            else:
                side.append(1)                          # second of a combined pair
        return pos, side

    def _brush_box(self, j):
        """(x0, width) of the brush rectangle for drawn column j."""
        x, sd = self.col_pos[j], self.col_side[j]
        if sd < 0:
            return x - 0.14, 0.14
        if sd > 0:
            return x, 0.14
        return x - 0.07, 0.14

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

    def build_segments(self, Y, xpos=None):
        n, m = Y.shape
        xpos = np.arange(m, dtype=float) if xpos is None else np.asarray(xpos, dtype=float)
        return np.stack([np.broadcast_to(xpos, (n, m)), Y], axis=2)

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
        self.drag = None

        if len(cols) < 2:
            self.ax = None
            self.fig.text(0.5, 0.5, "Show at least two axes", ha="center", va="center",
                          fontsize=14, color="gray")
            self.status.set("")
            self.canvas.draw_idle()
            return

        # In the paired layouts axes are used in pairs: with an odd number of axes,
        # duplicate the last axis (same name, same values) so that every axis has a
        # partner. Plain parallel coordinates (button 1) never duplicates.
        dup_last = self.mode != MODE_PC and len(cols) % 2 == 1
        if dup_last:
            cols = cols + [cols[-1]]
        self.drawn_cols = cols

        Y, lo, span = self.compute_layout(cols)
        rgba, handles, mappable = self.build_colors()
        m = len(cols)
        pos, side = self.layout_positions(m)
        self.col_pos, self.col_side = pos, side
        segments = self.build_segments(Y, pos)
        mask = self.brush_mask(Y, cols)

        # Figure layout (manual so legend / colourbar never collide with the plot)
        bottom = 0.14 if handles else 0.06
        width = 0.86 if mappable else 0.94
        ax = self.fig.add_axes([0.03, bottom, width, 0.80 - (bottom - 0.06)])
        self.ax = ax

        lw = float(self.linewidth.get())
        sel_colors = rgba[mask]
        if self.brushes:
            sel_colors = sel_colors.copy()
            sel_colors[:, 3] = np.maximum(sel_colors[:, 3], 0.8)
        if self.mode != MODE_PPC:       # paired parallel coordinates hides the original lines
            if (~mask).any():
                ax.add_collection(LineCollection(segments[~mask], colors=(0.7, 0.7, 0.7, 0.15),
                                                 linewidths=lw * 0.8, zorder=1))
            ax.add_collection(LineCollection(segments[mask], colors=sel_colors,
                                             linewidths=lw * (1.2 if self.brushes else 1.0), zorder=2))

        # Axes, labels and ticks (a combined axis shows the first column's ticks on its
        # left and the second column's ticks on its right)
        for x0 in sorted(set(pos)):
            ax.axvline(x0, color="black", lw=1.6, zorder=4)
        titles = {}
        for j, c in enumerate(cols):
            x0, sd = pos[j], side[j]
            name = c + (" dup" if dup_last and j == m - 1 else "")
            name += " ↕" if c in self.inverted else ""
            titles.setdefault(x0, []).append(name)
            for t in np.linspace(0, 1, N_TICKS):
                frac = 1 - t if c in self.inverted else t
                value = lo[j] + frac * span[j]
                if sd < 0:
                    ax.plot([x0 - 0.05, x0], [t, t], color="black", lw=1, zorder=4)
                    tx, ha = x0 - 0.04, "right"
                elif sd > 0:
                    ax.plot([x0, x0 + 0.05], [t, t], color="black", lw=1, zorder=4)
                    tx, ha = x0 + 0.04, "left"
                else:
                    ax.plot([x0 - 0.025, x0 + 0.025], [t, t], color="black", lw=1, zorder=4)
                    tx, ha = x0 + 0.04, "left"
                ax.text(tx, t, f"{value:.3g}", ha=ha, va="center", fontsize=7,
                        color="#333", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
        for x0, names in titles.items():
            ax.text(x0, 1.07, "  |  ".join(names), ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

        # Fans and intersection points, repeated for each consecutive pair of visible axes:
        # (axis 0, axis 1), (axis 2, axis 3), (axis 4, axis 5), ...
        # For a pair (j, j+1) with normalised values a (left axis) and b (right axis):
        #   fan A: (j, a)   -> (j+1, 0)  =>  y = a * (1 - t)      t = x - j
        #   fan B: (j+1, b) -> (j, 0)    =>  y = b * t
        #   a(1 - t) = b t               =>  t = a / (a + b),  y = a b / (a + b)
        # A trailing unpaired axis (odd number of visible axes) gets no fan.
        n_rows = len(Y)
        dots = []                       # per pair: (x array, y array) of intersection points
        for j in range(0, m - 1, 2):
            a, b = Y[:, j], Y[:, j + 1]

            if self.mode == MODE_RPPC:
                fan = np.zeros((n_rows, 2, 2))
                fan[:, 0, 0] = pos[j]        # left axis x
                fan[:, 0, 1] = a            # each row's value on the left axis
                fan[:, 1, 0] = pos[j + 1]    # right axis x
                fan[:, 1, 1] = 0.0          # origin of the right axis
                if (~mask).any():
                    ax.add_collection(LineCollection(fan[~mask], colors=(0.7, 0.7, 0.7, 0.15),
                                                     linewidths=lw * 0.8, linestyles="dashed", zorder=1))
                ax.add_collection(LineCollection(fan[mask], colors=sel_colors,
                                                 linewidths=lw, linestyles="dashed", zorder=2))

            if self.mode == MODE_RPPC:
                rev = np.zeros((n_rows, 2, 2))
                rev[:, 0, 0] = pos[j + 1]    # right axis x
                rev[:, 0, 1] = b            # each row's value on the right axis
                rev[:, 1, 0] = pos[j]        # left axis x
                rev[:, 1, 1] = 0.0          # origin of the left axis
                if (~mask).any():
                    ax.add_collection(LineCollection(rev[~mask], colors=(0.7, 0.7, 0.7, 0.15),
                                                     linewidths=lw * 0.8, linestyles="dashdot", zorder=1))
                ax.add_collection(LineCollection(rev[mask], colors=sel_colors,
                                                 linewidths=lw, linestyles="dashdot", zorder=2))

            denom = a + b
            ok = denom > 0              # a = b = 0: both lines lie on the base, no single point
            safe = np.where(ok, denom, 1.0)
            px = np.where(ok, pos[j] + (pos[j + 1] - pos[j]) * a / safe, np.nan)
            py = np.where(ok, a * b / safe, np.nan)
            dots.append((px, py))

            if self.mode != MODE_PC:    # no intersection dots in plain parallel coordinates
                sel = mask & ok
                uns = ~mask & ok
                if uns.any():
                    ax.scatter(px[uns], py[uns], s=14, color=(0.7, 0.7, 0.7, 0.3),
                               edgecolors="none", zorder=3)
                pc = sel_colors[ok[mask]].copy()
                pc[:, 3] = np.maximum(pc[:, 3], 0.9)
                ax.scatter(px[sel], py[sel], s=26, color=pc, edgecolors="black",
                           linewidths=0.5, zorder=5)

        # Paired parallel coordinates: join each row's intersection dots across pairs,
        # i.e. dot of (x1,x2) -> dot of (x3,x4) -> dot of (x5,x6) -> ...
        if self.mode == MODE_PPC and len(dots) > 1:
            row_colors = rgba.copy()
            if self.brushes:
                row_colors[:, 3] = np.maximum(row_colors[:, 3], 0.8)
            for (x0, y0), (x1, y1) in zip(dots, dots[1:]):
                link = np.stack([np.column_stack([x0, y0]), np.column_stack([x1, y1])], axis=1)
                good = np.isfinite(link).all(axis=(1, 2))   # skip rows without a dot
                uns = ~mask & good
                sel = mask & good
                if uns.any():
                    ax.add_collection(LineCollection(link[uns], colors=(0.7, 0.7, 0.7, 0.15),
                                                     linewidths=lw * 0.8, zorder=3))
                if sel.any():
                    ax.add_collection(LineCollection(link[sel], colors=row_colors[sel],
                                                     linewidths=lw * 1.4, zorder=4.5))

        # Brush rectangles
        for j, c in enumerate(cols):
            if c in self.brushes:
                b_lo, b_hi = self.brushes[c]
                bx, bw = self._brush_box(j)
                ax.add_patch(Rectangle((bx, b_lo), bw, b_hi - b_lo, fc="#ff7f0e",
                                       ec="#d35400", alpha=0.35, lw=1.2, zorder=6))

        ax.set_xlim(-0.35, max(pos) + 0.55)
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
        x = event.xdata
        for j, (p, sd) in enumerate(zip(self.col_pos, self.col_side)):
            if sd < 0:
                lo_x, hi_x = p - BRUSH_TOLERANCE, p
            elif sd > 0:
                lo_x, hi_x = p, p + BRUSH_TOLERANCE
            else:
                lo_x, hi_x = p - BRUSH_TOLERANCE, p + BRUSH_TOLERANCE
            if lo_x <= x <= hi_x:
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
            bx, bw = self._brush_box(j)
            patch = Rectangle((bx, y), bw, 0, fc="#ff7f0e", ec="#d35400",
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
