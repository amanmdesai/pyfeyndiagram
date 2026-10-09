from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

import numpy as np

from .generate import Diagram
from .model import FERMIONS, anti

WIDTH = 2.4
ROW = 1.3
EPS = 1e-6


class Line(NamedTuple):
    a: str
    b: str
    pdg: int
    external: bool


@dataclass
class Layout:
    diagram: Diagram
    pos: dict
    lines: list
    externals: dict

    @property
    def ymin(self) -> float:
        return min(y for _, y in self.pos.values())

    @property
    def ymax(self) -> float:
        return max(y for _, y in self.pos.values())


def label_side(p, q) -> int:
    """+1 if the label belongs on the left of p->q, -1 if on the right (prefers up, then right)."""
    nx, ny = -(q[1] - p[1]), q[0] - p[0]
    if ny < -EPS or (abs(ny) <= EPS and nx < 0):
        return -1
    return 1


def layout(diagram: Diagram) -> Layout:
    n = len(diagram.legs)
    if diagram.n_in not in (1, 2):
        raise ValueError("only 1 or 2 incoming particles are supported")

    adj = defaultdict(list)
    for e in diagram.edges:
        adj[e.u].append(e.v)
        adj[e.v].append(e.u)

    @lru_cache(maxsize=None)
    def leaves(node, parent) -> tuple:
        if node.startswith("x"):
            return (int(node[1:]),)
        found = []
        for nb in sorted((m for m in adj[node] if m != parent), key=lambda m: min(leaves(m, node))):
            found.extend(leaves(nb, node))
        return tuple(found)

    def hanging(vertex, skip):
        return sorted((m for m in adj[vertex] if m not in skip), key=lambda m: min(leaves(m, vertex)))

    if diagram.n_in == 1:
        out_order = list(leaves(adj["x0"][0], "x0"))
    else:
        prev = {"x0": None}
        queue = ["x0"]
        while queue:
            cur = queue.pop(0)
            for nb in adj[cur]:
                if nb not in prev:
                    prev[nb] = cur
                    queue.append(nb)
        path = []
        cur = "x1"
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        out_order = []
        for k in range(1, len(path) - 1):
            for nb in hanging(path[k], (path[k - 1], path[k + 1])):
                out_order.extend(leaves(nb, path[k]))

    k = len(out_order)
    pos = {}
    for j, idx in enumerate(out_order):
        pos[f"x{idx}"] = (WIDTH, ((k - 1) / 2 - j) * ROW)
    if diagram.n_in == 1:
        pos["x0"] = (0.0, 0.0)
    else:
        h = max(ROW / 2, (k - 1) / 2 * ROW)
        pos["x0"] = (0.0, h)
        pos["x1"] = (0.0, -h)

    internal = sorted(v for v in adj if v.startswith("v"))
    index = {v: i for i, v in enumerate(internal)}
    m = len(internal)
    A = np.zeros((m, m))
    bx = np.zeros(m)
    by = np.zeros(m)
    for v in internal:
        i = index[v]
        A[i, i] = len(adj[v])
        for nb in adj[v]:
            if nb in index:
                A[i, index[nb]] -= 1
            else:
                bx[i] += pos[nb][0]
                by[i] += pos[nb][1]
    xs = np.linalg.solve(A, bx)
    ys = np.linalg.solve(A, by)
    for v in internal:
        pos[v] = (float(xs[index[v]]), float(ys[index[v]]))

    groups = defaultdict(list)
    for idx in out_order:
        groups[diagram.external_pdg(idx)].append(idx)
    for members in groups.values():
        slots = sorted((pos[f"x{i}"][1] for i in members), reverse=True)
        for i, y in zip(sorted(members), slots):
            pos[f"x{i}"] = (WIDTH, y)

    lines = []
    for e in diagram.edges:
        external = e.u.startswith("x")
        dx = pos[e.v][0] - pos[e.u][0]
        if dx < -EPS or (abs(dx) <= EPS and e.pdg < 0 and abs(e.pdg) in FERMIONS):
            lines.append(Line(e.v, e.u, anti(e.pdg), external))
        else:
            lines.append(Line(e.u, e.v, e.pdg, external))

    externals = {f"x{i}": (diagram.external_pdg(i), i < diagram.n_in) for i in range(n)}
    return Layout(diagram, pos, lines, externals)
