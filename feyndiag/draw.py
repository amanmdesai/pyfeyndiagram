from __future__ import annotations

import numpy as np
from matplotlib import rc_context
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from .layout import WIDTH, label_side, layout
from .model import LATEX, kind

WAVE_AMP = 0.05
WAVELENGTH = 0.2
GLUON_R = 0.06
LABEL_OFFSET = 0.17
PAGE_SIZE = (8.27, 11.69)

_RC = {"mathtext.fontset": "cm", "font.family": "serif", "pdf.fonttype": 42}


def _line_points(style: str, p, q) -> np.ndarray:
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    length = float(np.hypot(*d))
    normal = np.array([-d[1], d[0]]) / length
    if style in ("photon", "boson"):
        t = np.linspace(0, 1, max(80, int(length * 120)))
        k = max(1, round(length / WAVELENGTH))
        return p + np.outer(t, d) + np.outer(WAVE_AMP * np.sin(2 * np.pi * k * t), normal)
    if style == "gluon":
        k = max(2, round(length * 1.4 / (2 * np.pi * GLUON_R)))
        c = length / (2 * np.pi * k)
        theta = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi * k, 60 * k)
        along = c * (theta - np.pi / 2) + GLUON_R * (1 - np.sin(theta))
        across = -GLUON_R * np.cos(theta)
        return p + np.outer(along / length, d) + np.outer(across, normal)
    t = np.linspace(0, 1, 2)
    return p + np.outer(t, d)


def draw_diagram(ax, lay) -> None:
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.75, WIDTH + 0.75)
    ax.set_ylim(lay.ymin - 0.5, lay.ymax + 0.5)

    for line in lay.lines:
        p, q = lay.pos[line.a], lay.pos[line.b]
        style = kind(line.pdg)
        pts = _line_points(style, p, q)
        linestyle = (0, (4, 3)) if style == "scalar" else "-"
        ax.plot(pts[:, 0], pts[:, 1], color="k", lw=1.1, ls=linestyle, solid_capstyle="round")
        d = np.asarray(q) - np.asarray(p)
        unit = d / np.hypot(*d)
        mid = (np.asarray(p) + np.asarray(q)) / 2
        if style == "fermion":
            tip = mid + unit * 0.07 * np.sign(line.pdg)
            tail = mid - unit * 0.07 * np.sign(line.pdg)
            ax.annotate(
                "", xy=tip, xytext=tail,
                arrowprops=dict(arrowstyle="-|>", color="k", lw=0, mutation_scale=11, shrinkA=0, shrinkB=0),
            )
        if not line.external:
            normal = label_side(p, q) * np.array([-unit[1], unit[0]])
            offset = LABEL_OFFSET + (GLUON_R if style == "gluon" else 0)
            x, y = mid + normal * offset
            ax.text(x, y, f"${LATEX[line.pdg]}$", ha="center", va="center", fontsize=11)

    for node, (pdg, incoming) in lay.externals.items():
        x, y = lay.pos[node]
        ax.text(
            x + (-0.08 if incoming else 0.08), y, f"${LATEX[pdg]}$",
            ha="right" if incoming else "left", va="center", fontsize=12,
        )


def _page_grid(layouts) -> tuple:
    height = layouts[0].ymax - layouts[0].ymin + 1.0
    cols = 2 if height > 3.0 or len(layouts) <= 4 else 3
    return cols, 4


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
                    f"  (page {page + 1}/{pages})",
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
