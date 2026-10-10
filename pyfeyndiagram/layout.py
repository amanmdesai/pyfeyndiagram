from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import NamedTuple

import numpy as np

from .generate import Diagram

MIN_WIDTH = 2.4
VERTEX_SPACING = 1.2
ROW = 1.3
EPS = 1e-6
NO_LEAF = 10 ** 6
RING_RADIUS = 0.55
STUB_LENGTH = 1.1
LOOP_SPACING = 1.2
LOOP_ROW = 1.6
RING_WEIGHT = 0.5
LOOP_BEND = 0.4
LOOP_RADIUS = 0.4


class Line(NamedTuple):
    a: str
    b: str
    pdg: int
    external: bool
    bend: float = 0.0
    side: int = 0


@dataclass
class Layout:
    """Vertex positions and lines. ``bend`` on a Line is the arc height as a fraction of its
    chord (positive bulges left of a->b); on a self-loop (a == b) it is the direction of the
    loop's axis in radians."""

    diagram: Diagram
    pos: dict
    lines: list
    externals: dict
    width: float = MIN_WIDTH
    reach: list = field(default_factory=list)

    def _coords(self, axis: int) -> list:
        return [p[axis] for p in list(self.pos.values()) + self.reach]

    @property
    def xmin(self) -> float:
        return min(self._coords(0))

    @property
    def xmax(self) -> float:
        return max(self._coords(0))

    @property
    def ymin(self) -> float:
        return min(self._coords(1))

    @property
    def ymax(self) -> float:
        return max(self._coords(1))


def label_side(p, q) -> int:
    """+1 if the label belongs on the left of p->q, -1 if on the right (prefers up, then right)."""
    nx, ny = -(q[1] - p[1]), q[0] - p[0]
    if ny < -EPS or (abs(ny) <= EPS and nx < 0):
        return -1
    return 1


def _out_order(adj: dict, n_in: int) -> list:
    """Order of the outgoing legs from a tree, so that the drawing needs few crossings."""

    @lru_cache(maxsize=None)
    def leaves(node, parent) -> tuple:
        if node.startswith("x"):
            return (int(node[1:]),)
        found = []
        for nb in sorted((m for m in adj[node] if m != parent), key=lambda m: min(leaves(m, node), default=NO_LEAF)):
            found.extend(leaves(nb, node))
        return tuple(found)

    def hanging(vertex, skip):
        return sorted((m for m in adj[vertex] if m not in skip), key=lambda m: min(leaves(m, vertex), default=NO_LEAF))

    if n_in == 1:
        return list(leaves(adj["x0"][0], "x0"))
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
    return out_order


def _embed(
    diagram: Diagram, adj: dict, out_order: list, spacing: float = VERTEX_SPACING, row: float = ROW,
    weights: dict = None, resolve: bool = False, fixed: dict = None,
) -> Layout:
    """Pin the external legs and put every internal vertex at the mean of its neighbours."""
    model = diagram.model
    n = len(diagram.legs)

    def farthest(start):
        dist = {start: 0}
        queue = [start]
        while queue:
            cur = queue.pop(0)
            for nb in adj[cur]:
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    queue.append(nb)
        node = max(dist, key=dist.get)
        return node, dist[node]

    diameter = farthest(farthest("x0")[0])[1]
    width = max(MIN_WIDTH, spacing * (diameter - 1))

    k = len(out_order)
    pos = {}
    for j, idx in enumerate(out_order):
        pos[f"x{idx}"] = (width, ((k - 1) / 2 - j) * row)
    if diagram.n_in == 1:
        pos["x0"] = (0.0, 0.0)
    else:
        h = max(row / 2, (k - 1) / 2 * row)
        pos["x0"] = (0.0, h)
        pos["x1"] = (0.0, -h)

    fixed = fixed or {}
    pos.update(fixed)
    internal = sorted(v for v in adj if v.startswith("v") and v not in fixed)
    index = {v: i for i, v in enumerate(internal)}
    m = len(internal)
    weights = weights or {}

    def solve():
        A = np.zeros((m, m))
        bx = np.zeros(m)
        by = np.zeros(m)
        for v in internal:
            i = index[v]
            for nb in adj[v]:
                w = weights.get(frozenset((v, nb)), 1.0)
                A[i, i] += w
                if nb in index:
                    A[i, index[nb]] -= w
                else:
                    bx[i] += w * pos[nb][0]
                    by[i] += w * pos[nb][1]
        if not m:
            return
        xs = np.linalg.solve(A, bx)
        ys = np.linalg.solve(A, by)
        for v in internal:
            pos[v] = (float(xs[index[v]]), float(ys[index[v]]))

    solve()

    groups = defaultdict(list)
    for idx in out_order:
        groups[diagram.external_pdg(idx)].append(idx)
    for members in groups.values():
        slots = sorted((pos[f"x{i}"][1] for i in members), reverse=True)
        for i, y in zip(sorted(members), slots):
            pos[f"x{i}"] = (width, y)
    if resolve:
        solve()

    lines = []
    for e in diagram.edges:
        external = e.u.startswith("x")
        dx = pos[e.v][0] - pos[e.u][0]
        if dx < -EPS or (abs(dx) <= EPS and e.pdg < 0 and model.has_arrow(e.pdg)):
            lines.append(Line(e.v, e.u, model.anti(e.pdg), external))
        else:
            lines.append(Line(e.u, e.v, e.pdg, external))

    externals = {f"x{i}": (diagram.external_pdg(i), i < diagram.n_in) for i in range(n)}
    return Layout(diagram, pos, lines, externals, width)


def _reaches(lay: Layout, adj: dict) -> None:
    """Bend parallel lines apart and orient self-loops away from their neighbours."""
    pairs = defaultdict(list)
    for i, line in enumerate(lay.lines):
        pairs[frozenset((line.a, line.b))].append(i)
    for key, members in pairs.items():
        if len(key) == 1:
            (v,) = key
            px, py = lay.pos[v]
            neighbours = adj[v]
            mx = sum(lay.pos[nb][0] for nb in neighbours) / len(neighbours) - px
            my = sum(lay.pos[nb][1] for nb in neighbours) / len(neighbours) - py
            angle = math.atan2(-my, -mx) if math.hypot(mx, my) > EPS else math.pi / 2
            for i in members:
                line = lay.lines[i]
                lay.lines[i] = line._replace(bend=angle)
                cx, cy = px + LOOP_RADIUS * math.cos(angle), py + LOOP_RADIUS * math.sin(angle)
                lay.reach += [(cx - LOOP_RADIUS, cy), (cx + LOOP_RADIUS, cy), (cx, cy - LOOP_RADIUS), (cx, cy + LOOP_RADIUS)]
        elif len(members) > 1:
            for rank, i in enumerate(members):
                line = lay.lines[i]
                height = LOOP_BEND * (1 if rank % 2 == 0 else -1) * (1 + rank // 2)
                sign = 1 if line.a == min(key) else -1
                lay.lines[i] = line._replace(bend=sign * height)
                p, q = lay.pos[min(key)], lay.pos[max(key)]
                mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
                chord = math.hypot(q[0] - p[0], q[1] - p[1])
                normal = (-(q[1] - p[1]) / chord, (q[0] - p[0]) / chord) if chord > EPS else (0.0, 1.0)
                lay.reach.append((mid[0] + normal[0] * height * chord, mid[1] + normal[1] * height * chord))


def _pull_stubs(lay: Layout, adj: dict) -> None:
    """Move a dead-end vertex (a tadpole's loop) off its single neighbour, away from the rest."""
    for v in [v for v in adj if v.startswith("v") and len(adj[v]) == 1]:
        (u,) = adj[v]
        others = [lay.pos[w] for w in adj[u] if w != v]
        ux, uy = lay.pos[u]
        if others:
            dx = ux - sum(p[0] for p in others) / len(others)
            dy = uy - sum(p[1] for p in others) / len(others)
        else:
            dx = dy = 0.0
        norm = math.hypot(dx, dy)
        dx, dy = (dx / norm, dy / norm) if norm > EPS else (0.0, 1.0)
        lay.pos[v] = (ux + STUB_LENGTH * dx, uy + STUB_LENGTH * dy)


def _label_outside(lay: Layout, ring_vertices: set) -> None:
    """Put the labels of a loop with three or more vertices on its outer side."""
    if not ring_vertices:
        return
    cx = sum(lay.pos[v][0] for v in ring_vertices) / len(ring_vertices)
    cy = sum(lay.pos[v][1] for v in ring_vertices) / len(ring_vertices)
    for i, line in enumerate(lay.lines):
        if line.a in ring_vertices and line.b in ring_vertices:
            (ax, ay), (bx, by) = lay.pos[line.a], lay.pos[line.b]
            outward = -(by - ay) * ((ax + bx) / 2 - cx) + (bx - ax) * ((ay + by) / 2 - cy)
            lay.lines[i] = line._replace(side=1 if outward > 0 else -1)


def _cycle(ring: list) -> list:
    """The vertices of the loop in order around it."""
    nbrs = defaultdict(set)
    for e in ring:
        nbrs[e.u].add(e.v)
        nbrs[e.v].add(e.u)
    order = [min(nbrs)]
    while len(order) < len(nbrs):
        order.append(min(w for w in nbrs[order[-1]] if w not in order))
    return order


def _degenerate(lay: Layout, cycle: list) -> bool:
    pts = [lay.pos[v] for v in cycle]
    if len(pts) < 2:
        return False
    if min(math.dist(p, q) for i, p in enumerate(pts) for q in pts[:i]) < 0.45:
        return True
    if len(pts) == 3:
        (ax, ay), (bx, by), (cx, cy) = pts
        return abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) / 2 < 0.2
    return False


def _length(lay: Layout, adj: dict) -> float:
    return sum(math.dist(lay.pos[a], lay.pos[b]) for a in adj for b in adj[a] if a < b)


def _regularise(diagram, simple, out_order, weights, cycle, lay) -> Layout:
    """Replace a collapsed loop by a regular polygon, trying every orientation."""
    k = len(cycle)
    cx = sum(lay.pos[v][0] for v in cycle) / k
    cy = sum(lay.pos[v][1] for v in cycle) / k
    radius = RING_RADIUS + 0.1 * k
    best = None
    for sign in (1, -1):
        for step in range(12):
            alpha = step * math.pi / 6
            fixed = {
                v: (cx + radius * math.cos(alpha + sign * 2 * math.pi * i / k),
                    cy + radius * math.sin(alpha + sign * 2 * math.pi * i / k))
                for i, v in enumerate(cycle)
            }
            cand = _embed(
                diagram, simple, out_order, VERTEX_SPACING * LOOP_SPACING, ROW * LOOP_ROW,
                weights, resolve=True, fixed=fixed,
            )
            _pull_stubs(cand, simple)
            score = 10 * _crossings(cand) + _length(cand, simple)
            if best is None or score < best[0] - 1e-9:
                best = (score, cand)
    return best[1]


def _crossings(lay: Layout) -> int:
    segments = []
    seen = set()
    for line in lay.lines:
        key = frozenset((line.a, line.b))
        if len(key) == 2 and key not in seen:
            seen.add(key)
            segments.append((line.a, line.b))

    def side(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])

    count = 0
    for i, (a, b) in enumerate(segments):
        for c, d in segments[:i]:
            if len({a, b, c, d}) < 4:
                continue
            pa, pb, pc, pd = lay.pos[a], lay.pos[b], lay.pos[c], lay.pos[d]
            if side(pa, pb, pc) * side(pa, pb, pd) < -EPS and side(pc, pd, pa) * side(pc, pd, pb) < -EPS:
                count += 1
    return count


def _connected_without(edges: list, skip: int) -> bool:
    adj = defaultdict(list)
    for i, e in enumerate(edges):
        if i != skip:
            adj[e.u].append(e.v)
            adj[e.v].append(e.u)
    seen, stack = {edges[skip].u}, [edges[skip].u]
    while stack:
        for nb in adj[stack.pop()]:
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return edges[skip].v in seen


def _layout_loop(diagram: Diagram) -> Layout:
    """Try each way of cutting the loop open to order the outer legs; keep the fewest crossings."""
    edges = diagram.edges
    simple = defaultdict(list)
    for e in edges:
        if e.u != e.v and e.v not in simple[e.u]:
            simple[e.u].append(e.v)
            simple[e.v].append(e.u)

    ring = [e for i, e in enumerate(edges) if not e.u.startswith("x") and e.u != e.v and _connected_without(edges, i)]
    ring_vertices = {v for e in ring for v in (e.u, e.v)}
    cycle = _cycle(ring) if ring else []
    weights = {}
    if len(ring_vertices) > 2:
        weights = {frozenset((e.u, e.v)): RING_WEIGHT for e in ring}

    best = None
    tried = set()
    for cut, e in enumerate(edges):
        if e.u.startswith("x") or not _connected_without(edges, cut):
            continue
        tree = defaultdict(list)
        for i, f in enumerate(edges):
            if i != cut and f.u != f.v:
                tree[f.u].append(f.v)
                tree[f.v].append(f.u)
        out_order = _out_order(tree, diagram.n_in)
        if tuple(out_order) in tried:
            continue
        tried.add(tuple(out_order))
        lay = _embed(diagram, simple, out_order, VERTEX_SPACING * LOOP_SPACING, ROW * LOOP_ROW, weights, resolve=True)
        _pull_stubs(lay, simple)
        if _degenerate(lay, cycle):
            lay = _regularise(diagram, simple, out_order, weights, cycle, lay)
        _reaches(lay, simple)
        _label_outside(lay, ring_vertices if weights else set())
        score = _crossings(lay)
        if best is None or score < best[0]:
            best = (score, lay)
    return best[1]


def layout(diagram: Diagram) -> Layout:
    if diagram.n_in not in (1, 2):
        raise ValueError("only 1 or 2 incoming particles are supported")
    if diagram.loops:
        return _layout_loop(diagram)

    adj = defaultdict(list)
    for e in diagram.edges:
        adj[e.u].append(e.v)
        adj[e.v].append(e.u)
    return _embed(diagram, adj, _out_order(adj, diagram.n_in))
