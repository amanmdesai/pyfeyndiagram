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


def _build_diagram(top: Node, legs: tuple, n_in: int, model: Model) -> Diagram:
    last = len(legs) - 1
    edges = []
    counter = [0]

    def attach(node: Node, vid: str):
        for child in sorted(node.children, key=lambda c: _leaves(c)[0]):
            if isinstance(child, Leaf):
                edges.append(Edge(f"x{child.index}", vid, child.label))
            else:
                counter[0] += 1
                cid = f"v{counter[0]}"
                edges.append(Edge(cid, vid, child.label))
                attach(child, cid)

    attach(top, "v0")
    edges.append(Edge(f"x{last}", "v0", legs[last]))
    return Diagram(legs, n_in, edges, top.qcd, top.qed, model)


def _has_resonances(diagram: Diagram, resonances: tuple) -> bool:
    anti = diagram.model.anti
    adj = defaultdict(list)
    for e in diagram.edges:
        adj[e.u].append(e.v)
        adj[e.v].append(e.u)

    def side(start: str, blocked: str) -> frozenset:
        seen, stack, found = {start, blocked}, [start], set()
        while stack:
            node = stack.pop()
            if node.startswith("x"):
                found.add(int(node[1:]))
            for nb in adj[node]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return frozenset(found)

    cuts = [(side(e.u, e.v), e.pdg) for e in diagram.edges if e.u.startswith("v")]
    everything = frozenset(range(len(diagram.legs)))
    for parent, leaves in resonances:
        wanted = frozenset(leaves)
        if not any(
            (below == wanted and pdg == anti(parent)) or (below == everything - wanted and pdg == parent)
            for below, pdg in cuts
        ):
            return False
    return True


def generate(process: Process, model: Model = None) -> list:
    """All tree-level diagrams of a process, as a list of Diagram."""
    model = model or process.model
    legs = process.legs
    n = len(legs)
    last = n - 1
    full = (1 << last) - 1
    max_qcd = float("inf") if process.max_qcd is None else process.max_qcd
    max_qed = float("inf") if process.max_qed is None else process.max_qed

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
                    elif abs(missing) in process.excluded:
                        continue
                    qcd = vqcd + sum(_orders(s)[0] for _, s in combo)
                    qed = vqed + sum(_orders(s)[1] for _, s in combo)
                    if qcd > max_qcd or qed > max_qed:
                        continue
                    node = Node(model.anti(missing), frozenset(s for _, s in combo), qcd, qed)
                    found[node.label].add(node)
        currents[mask] = found

    tops = [s for ss in currents[full].values() for s in ss]
    tops.sort(key=_signature)
    diagrams = [_build_diagram(top, legs, len(process.incoming), model) for top in tops]
    if process.resonances:
        diagrams = [d for d in diagrams if _has_resonances(d, process.resonances)]
    return diagrams


def generate_all(text: str, model: Model = None, max_qcd=None, max_qed=None) -> list:
    """Parse a process string and return [(Process, [Diagram])] for each subprocess with diagrams."""
    model = model or Model()
    results = []
    for process in parse_process(text, max_qcd, max_qed, model):
        diagrams = generate(process, model)
        if diagrams:
            results.append((process, diagrams))
    return results
