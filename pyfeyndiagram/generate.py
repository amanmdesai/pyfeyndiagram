from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import NamedTuple

from .model import Model
from .process import Process, parse_process


class Edge(NamedTuple):
    u: str
    v: str
    pdg: int


@dataclass
class Diagram:
    legs: tuple
    n_in: int
    edges: list
    qcd: int
    qed: int
    model: Model
    loops: int = 0

    def external_pdg(self, index: int) -> int:
        return self.legs[index] if index < self.n_in else self.model.anti(self.legs[index])


class Leaf(NamedTuple):
    index: int
    label: int


class Node(NamedTuple):
    label: int
    children: frozenset
    qcd: int
    qed: int


def _orders(struct):
    return (0, 0) if isinstance(struct, Leaf) else (struct.qcd, struct.qed)


def _leaves(struct) -> tuple:
    if isinstance(struct, Leaf):
        return (struct.index,)
    return tuple(sorted(i for c in struct.children for i in _leaves(c)))


def _signature(struct) -> list:
    if isinstance(struct, Leaf):
        return []
    sig = [(len(_leaves(struct)), _leaves(struct), struct.label)]
    for child in struct.children:
        sig += _signature(child)
    return sorted(sig)


def _partitions(mask: int):
    """Unordered splits of a bitmask into 2 or 3 non-empty blocks."""
    low = mask & -mask
    rest = mask ^ low
    sub = rest
    while True:
        first = low | sub
        remaining = mask ^ first
        if remaining:
            yield (first, remaining)
            if remaining & (remaining - 1):
                low2 = remaining & -remaining
                rest2 = remaining ^ low2
                sub2 = rest2
                while True:
                    second = low2 | sub2
                    third = remaining ^ second
                    if third:
                        yield (first, second, third)
                    if sub2 == 0:
                        break
                    sub2 = (sub2 - 1) & rest2
        if sub == 0:
            break
        sub = (sub - 1) & rest


def _attach(children, vid: str, edges: list, counter: list) -> None:
    for child in sorted(children, key=lambda c: _leaves(c)[0]):
        if isinstance(child, Leaf):
            edges.append(Edge(f"x{child.index}", vid, child.label))
        else:
            counter[0] += 1
            cid = f"v{counter[0]}"
            edges.append(Edge(cid, vid, child.label))
            _attach(child.children, cid, edges, counter)


def _build_diagram(top: Node, legs: tuple, n_in: int, model: Model) -> Diagram:
    last = len(legs) - 1
    edges = []
    _attach(top.children, "v0", edges, [0])
    edges.append(Edge(f"x{last}", "v0", legs[last]))
    return Diagram(legs, n_in, edges, top.qcd, top.qed, model)


def _has_resonances(diagram: Diagram, resonances: tuple) -> bool:
    anti = diagram.model.anti
    edges = diagram.edges
    adj = defaultdict(list)
    for i, e in enumerate(edges):
        adj[e.u].append((e.v, i))
        adj[e.v].append((e.u, i))

    def side(i: int):
        """External legs on the u side of edge i, or None if the edge lies on a loop."""
        start = edges[i].u
        seen, stack, found = {start}, [start], set()
        while stack:
            node = stack.pop()
            if node.startswith("x"):
                found.add(int(node[1:]))
            for nb, j in adj[node]:
                if j != i and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return None if edges[i].v in seen else frozenset(found)

    cuts = [(side(i), e.pdg) for i, e in enumerate(edges) if e.u.startswith("v")]
    cuts = [(below, pdg) for below, pdg in cuts if below is not None]
    everything = frozenset(range(len(diagram.legs)))
    for parent, leaves in resonances:
        wanted = frozenset(leaves)
        if not any(
            (below == wanted and pdg == anti(parent)) or (below == everything - wanted and pdg == parent)
            for below, pdg in cuts
        ):
            return False
    return True


def _trees(legs: tuple, excluded: frozenset, max_qcd, max_qed, model: Model) -> list:
    """All tree structures with the last leg as root, for all-incoming external labels."""
    n = len(legs)
    last = n - 1
    full = (1 << last) - 1
    max_qcd = float("inf") if max_qcd is None else max_qcd
    max_qed = float("inf") if max_qed is None else max_qed

    currents = {0: {}}
    for i in range(last):
        currents[1 << i] = {legs[i]: {Leaf(i, legs[i])}}

    for mask in range(1, full + 1):
        if mask & (mask - 1) == 0:
            continue
        found = defaultdict(set)
        for blocks in _partitions(mask):
            pools = [[(lab, s) for lab, ss in currents[b].items() for s in ss] for b in blocks]
            for combo in product(*pools):
                labels = tuple(sorted(lab for lab, _ in combo))
                for missing, vqcd, vqed in model.completions.get(labels, ()):
                    if mask == full:
                        if missing != legs[last]:
                            continue
                    elif abs(missing) in excluded:
                        continue
                    qcd = vqcd + sum(_orders(s)[0] for _, s in combo)
                    qed = vqed + sum(_orders(s)[1] for _, s in combo)
                    if qcd > max_qcd or qed > max_qed:
                        continue
                    node = Node(model.anti(missing), frozenset(s for _, s in combo), qcd, qed)
                    found[node.label].add(node)
        currents[mask] = found

    return [s for ss in currents[full].values() for s in ss]


def _canon(struct) -> tuple:
    if isinstance(struct, Leaf):
        return (0, struct.index)
    return (1, struct.label, tuple(sorted(_canon(c) for c in struct.children)))


def _ring_key(ring: list) -> tuple:
    return tuple((tuple(_canon(h) for h in hangs), label) for hangs, label in ring)


def _canonical_ring(top: Node, leaf: int, phi: int, model: Model) -> list:
    """The loop of a tree whose root and leaf ``leaf`` are glued, in canonical orientation.

    The result lists, around the loop, the subtrees hanging off each loop vertex together with
    the label of the loop line leaving it; it is the minimum over rotations and reversal.
    """
    chain = [top]
    while not any(isinstance(c, Leaf) and c.index == leaf for c in chain[-1].children):
        chain.append(next(c for c in chain[-1].children if isinstance(c, Node) and leaf in _leaves(c)))

    ring = []
    for j in range(len(chain) - 1, -1, -1):
        skip = chain[j + 1] if j + 1 < len(chain) else Leaf(leaf, phi)
        hangs = tuple(sorted((c for c in chain[j].children if c != skip), key=_canon))
        ring.append((hangs, chain[j].label if j else phi))

    k = len(ring)
    reflected = [(ring[i][0], model.anti(ring[i - 1][1])) for i in range(k - 1, -1, -1)]
    candidates = [seq[r:] + seq[:r] for seq in (ring, reflected) for r in range(k)]
    return min(candidates, key=_ring_key)


def _build_loop_diagram(ring: list, qcd: int, qed: int, legs: tuple, n_in: int, model: Model) -> Diagram:
    k = len(ring)
    edges = [Edge(f"v{i}", f"v{(i + 1) % k}", label) for i, (_, label) in enumerate(ring)]
    counter = [k - 1]
    for i, (hangs, _) in enumerate(ring):
        _attach(hangs, f"v{i}", edges, counter)
    return Diagram(legs, n_in, edges, qcd, qed, model, loops=1)


def _generate_loops(process: Process, model: Model) -> list:
    legs = process.legs
    n = len(legs)
    found = {}
    for phi in sorted(model.particles):
        if phi in process.excluded:
            continue
        trees = _trees(legs + (phi, model.anti(phi)), process.excluded, process.max_qcd, process.max_qed, model)
        for top in trees:
            ring = _canonical_ring(top, n, phi, model)
            hangs = [h for hs, _ in ring for h in hs]
            if len(hangs) == 1 and not process.tadpoles:
                continue
            if process.one_pi and any(isinstance(h, Node) for h in hangs):
                continue
            if not process.external_corrections and len(hangs) == 2 and any(isinstance(h, Leaf) for h in hangs):
                continue
            found.setdefault(_ring_key(ring), (ring, top.qcd, top.qed))
    order = sorted(found, key=lambda key: (len(key), key))
    return [
        _build_loop_diagram(*found[key], legs, len(process.incoming), model) for key in order
    ]


def generate(process: Process, model: Model = None) -> list:
    """All diagrams of a process (tree level, or one loop if ``process.loops``), as a list of Diagram."""
    model = model or process.model
    if process.loops:
        diagrams = _generate_loops(process, model)
    else:
        tops = _trees(process.legs, process.excluded, process.max_qcd, process.max_qed, model)
        tops.sort(key=_signature)
        diagrams = [_build_diagram(top, process.legs, len(process.incoming), model) for top in tops]
    if process.resonances:
        diagrams = [d for d in diagrams if _has_resonances(d, process.resonances)]
    return diagrams


def generate_all(
    text: str, model: Model = None, max_qcd=None, max_qed=None,
    loops: int = 0, one_pi: bool = False, tadpoles: bool = False, external_corrections: bool = True,
) -> list:
    """Parse a process string and return [(Process, [Diagram])] for each subprocess with diagrams."""
    model = model or Model()
    results = []
    for process in parse_process(text, max_qcd, max_qed, model, loops, one_pi, tadpoles, external_corrections):
        diagrams = generate(process, model)
        if diagrams:
            results.append((process, diagrams))
    return results
