from __future__ import annotations

import numpy as np
from matplotlib import rc_context
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from .layout import LOOP_RADIUS, label_side, layout

WAVE_AMP = 0.05
WAVELENGTH = 0.2
GLUON_R = 0.06
LABEL_OFFSET = 0.17
PAGE_SIZE = (8.27, 11.69)

_RC = {"mathtext.fontset": "cm", "font.family": "serif", "pdf.fonttype": 42}


def _shape(style: str, length: float):
    """Along-line and across-line offsets of a line of the given style and length."""
    if style in ("photon", "boson"):
        t = np.linspace(0, 1, max(80, int(length * 120)))
        k = max(1, round(length / WAVELENGTH))
        return t * length, WAVE_AMP * np.sin(2 * np.pi * k * t)
    if style == "gluon":
        k = max(2, round(length * 1.4 / (2 * np.pi * GLUON_R)))
        c = length / (2 * np.pi * k)
        theta = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi * k, 60 * k)
        return c * (theta - np.pi / 2) + GLUON_R * (1 - np.sin(theta)), -GLUON_R * np.cos(theta)
    t = np.linspace(0, 1, 2)
    return t * length, np.zeros(2)


def _line_points(style: str, p, q) -> np.ndarray:
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    length = float(np.hypot(*d))
    normal = np.array([-d[1], d[0]]) / length
    along, across = _shape(style, length)
    return p + np.outer(along / length, d) + np.outer(across, normal)


class _Arc:
    """A circular arc leaving ``start`` at ``heading``, turning by ``sweep`` (counterclockwise if positive)."""

    def __init__(self, start, heading: float, radius: float, sweep: float):
        self.start = np.asarray(start, float)
        self.heading = heading
        self.radius = radius
        self.sign = 1.0 if sweep >= 0 else -1.0
        self.length = radius * abs(sweep)

    def at(self, s):
        """Points, and left normals, at arclength s (array)."""
        h = self.heading + self.sign * np.asarray(s) / self.radius
        x = self.start[0] + self.sign * self.radius * (np.sin(h) - np.sin(self.heading))
        y = self.start[1] - self.sign * self.radius * (np.cos(h) - np.cos(self.heading))
        return np.stack([x, y], axis=-1), np.stack([-np.sin(h), np.cos(h)], axis=-1)


def _arc(p, q, bend: float) -> _Arc:
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    chord = float(np.hypot(*d))
    theta = 4 * np.arctan(2 * abs(bend))
    radius = chord / (2 * np.sin(theta / 2))
    sign = 1.0 if bend > 0 else -1.0
    return _Arc(p, float(np.arctan2(d[1], d[0])) + sign * theta / 2, radius, -sign * theta)


def _self_loop(p, axis: float, radius: float) -> _Arc:
    return _Arc(p, axis + np.pi / 2 + np.pi, radius, 2 * np.pi)


def _arc_points(style: str, arc: _Arc) -> np.ndarray:
    along, across = _shape(style, arc.length)
    if len(along) == 2:
        along, across = np.linspace(0, arc.length, 60), np.zeros(60)
    pts, normals = arc.at(along)
    return pts + across[:, None] * normals


def draw_diagram(ax, lay) -> None:
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(min(0.0, lay.xmin) - 0.75, max(lay.width, lay.xmax) + 0.75)
    ax.set_ylim(lay.ymin - 0.5, lay.ymax + 0.5)
    model = lay.diagram.model

    for line in lay.lines:
        p, q = lay.pos[line.a], lay.pos[line.b]
        style = model.kind(line.pdg)
        if line.a == line.b:
            arc = _self_loop(p, line.bend, LOOP_RADIUS)
        elif line.bend:
            arc = _arc(p, q, line.bend)
        else:
            arc = None
        pts = _line_points(style, p, q) if arc is None else _arc_points(style, arc)
        linestyle = {"scalar": (0, (4, 3)), "ghost": (0, (1, 2))}.get(style, "-")
        ax.plot(pts[:, 0], pts[:, 1], color="k", lw=1.1, ls=linestyle, solid_capstyle="round")
        if arc is None:
            d = np.asarray(q) - np.asarray(p)
            unit = d / np.hypot(*d)
            mid = (np.asarray(p) + np.asarray(q)) / 2
            normal = (line.side or label_side(p, q)) * np.array([-unit[1], unit[0]])
        else:
            (mid,), (left,) = arc.at([arc.length / 2])
            unit = np.array([left[1], -left[0]])
            normal = -arc.sign * left
        if model.has_arrow(line.pdg):
            tip = mid + unit * 0.07 * np.sign(line.pdg)
            tail = mid - unit * 0.07 * np.sign(line.pdg)
            ax.annotate(
                "", xy=tip, xytext=tail,
                arrowprops=dict(arrowstyle="-|>", color="k", lw=0, mutation_scale=11, shrinkA=0, shrinkB=0),
            )
        if not line.external:
            offset = LABEL_OFFSET + (GLUON_R if style == "gluon" else 0)
            x, y = mid + normal * offset
            ax.text(x, y, f"${model.latex(line.pdg)}$", ha="center", va="center", fontsize=11)

    for node, (pdg, incoming) in lay.externals.items():
        x, y = lay.pos[node]
        ax.text(
            x + (-0.08 if incoming else 0.08), y, f"${model.latex(pdg)}$",
            ha="right" if incoming else "left", va="center", fontsize=12,
        )


_GRIDS = [(3, 4), (3, 3), (2, 4), (2, 3), (3, 2), (2, 2), (1, 3), (1, 2), (1, 1)]
_AREA = (PAGE_SIZE[0] * 0.92, PAGE_SIZE[1] * 0.88)
_MIN_SCALE = 0.6


def _page_grid(layouts) -> tuple:
    lay = layouts[0]
    w = lay.width + 1.5
    h = lay.ymax - lay.ymin + 1.0
    cap = max(4, len(layouts) + 1)

    def scale(grid):
        cols, rows = grid
        return min(_AREA[0] / cols / w, _AREA[1] / rows / h)

    fitting = [g for g in _GRIDS if g[0] * g[1] <= cap]
    for grid in fitting:
        if scale(grid) >= _MIN_SCALE:
            return grid
    return max(fitting or _GRIDS[-1:], key=scale)


def write_pdf(results, path: str, title: str = None) -> None:
    """Render [(Process, [Diagram])] into a multi-page PDF."""
    with rc_context(_RC), PdfPages(path) as pdf:
        for process, diagrams in results:
            layouts = [layout(d) for d in diagrams]
            cols, rows = _page_grid(layouts)
            per_page = cols * rows
            pages = (len(layouts) + per_page - 1) // per_page
            for page in range(pages):
                fig = Figure(figsize=PAGE_SIZE)
                fig.text(
                    0.5, 0.965, f"${process.latex()}$", ha="center", va="center", fontsize=16,
                )
                fig.text(
                    0.5, 0.94, f"{len(diagrams)} diagram{'s' if len(diagrams) != 1 else ''}"
                    f"{', one loop' if process.loops else ''}  (page {page + 1}/{pages})",
                    ha="center", va="center", fontsize=9, color="0.4",
                )
                grid = fig.add_gridspec(
                    rows, cols, top=0.91, bottom=0.03, left=0.04, right=0.96, hspace=0.2, wspace=0.08,
                )
                chunk = layouts[page * per_page:(page + 1) * per_page]
                for j, lay in enumerate(chunk):
                    ax = fig.add_subplot(grid[j // cols, j % cols])
                    draw_diagram(ax, lay)
                    number = page * per_page + j + 1
                    ax.set_title(f"({number})", loc="left", fontsize=9)
                    ax.set_title(
                        f"QCD={lay.diagram.qcd}  QED={lay.diagram.qed}", loc="right", fontsize=7, color="0.4",
                    )
                pdf.savefig(fig)
