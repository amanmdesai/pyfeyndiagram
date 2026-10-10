from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import product
from typing import Optional

from .model import Model

_ORDER_RE = re.compile(r"\b(QCD|QED)\s*(?:<=|=)\s*(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class Process:
    incoming: tuple
    outgoing: tuple
    excluded: frozenset = frozenset()
    max_qcd: Optional[int] = None
    max_qed: Optional[int] = None
    production: Optional[tuple] = None
    decays: tuple = ()
    resonances: tuple = ()
    model: Optional[Model] = field(default=None, compare=False, repr=False)
    loops: int = 0
    one_pi: bool = False
    tadpoles: bool = False
    external_corrections: bool = True

    @property
    def legs(self) -> tuple:
        """All external labels in the all-incoming convention."""
        return tuple(self.incoming) + tuple(self.model.anti(p) for p in self.outgoing)

    def __str__(self) -> str:
        name = self.model.name
        final = self.outgoing if self.production is None else self.production
        text = " ".join(name(p) for p in self.incoming)
        text += " > " + " ".join(name(p) for p in final)
        for parent, kids in self.decays:
            text += f", {name(parent)} > " + " ".join(name(p) for p in kids)
        if self.excluded:
            text += " / " + " ".join(name(p) for p in sorted(self.excluded))
        if self.loops:
            text += f" [{self.loops} loop]"
        return text

    def latex(self) -> str:
        latex = self.model.latex
        final = self.outgoing if self.production is None else self.production
        ins = r"\,".join(latex(p) for p in self.incoming)
        outs = r"\,".join(latex(p) for p in final)
        text = rf"{ins} \to {outs}"
        for parent, kids in self.decays:
            text += rf",\; {latex(parent)} \to " + r"\,".join(latex(p) for p in kids)
        return text


class _Slot:
    def __init__(self, pdg: int):
        self.pdg = pdg
        self.kids = []

    def leaves(self) -> list:
        return [self] if not self.kids else [leaf for k in self.kids for leaf in k.leaves()]

    def canon(self) -> tuple:
        return (self.pdg, tuple(sorted(k.canon() for k in self.kids)))

    def decayed(self) -> list:
        return [self] + [s for k in self.kids for s in k.decayed()] if self.kids else []


def _split(statement: str, model: Model) -> tuple:
    if statement.count(">") != 1:
        raise ValueError(f"expected exactly one '>' in {statement.strip()!r}")
    lhs, rhs = statement.split(">")
    return [model.resolve(t) for t in lhs.split()], [model.resolve(t) for t in rhs.split()]


def _decay_options(statement: str, model: Model) -> list:
    parents, kids = _split(statement, model)
    if len(parents) != 1 or len(kids) < 2:
        raise ValueError(f"decay must be 'X > Y Z ...': {statement.strip()!r}")
    return [
        (parent, ks)
        for parent in parents[0]
        for ks in product(*kids)
        if model.charge3(parent) == sum(model.charge3(k) for k in ks)
    ]


def _apply_decays(outs: tuple, decays: tuple, model: Model) -> tuple:
    top = [_Slot(p) for p in outs]
    for parent, kids in decays:
        targets = [s for slot in top for s in slot.leaves() if s.pdg == parent]
        if not targets:
            raise ValueError(f"cannot decay {model.name(parent)}: not in the final state at that point")
        for slot in targets:
            slot.kids = [_Slot(k) for k in kids]
    return tuple(top)


def parse_process(
    text: str, max_qcd=None, max_qed=None, model: Model = None,
    loops: int = 0, one_pi: bool = False, tadpoles: bool = False, external_corrections: bool = True,
) -> list:
    """Parse a MadGraph-style process string into explicit subprocesses.

    Syntax: ``e+ e- > mu+ mu- [, X > Y Z ...] [/ z h] [QCD=n] [QED=n]``.
    Decay statements (optionally parenthesised, and chainable) restrict the
    diagrams to those where the decaying particle is a propagator.
    Multiparticle labels (``j``, ``p``, ``l+``, ``l-``, ``vl``, ``vl~`` in the SM) are expanded.

    ``loops=1`` asks for one-loop diagrams instead of tree diagrams. ``one_pi`` keeps only
    one-particle-irreducible ones, ``tadpoles`` includes tadpoles and ``external_corrections=False``
    drops self-energy insertions on external legs.
    """
    if loops not in (0, 1):
        raise ValueError("only 0 or 1 loops are supported")
    model = model or Model()
    for key, value in _ORDER_RE.findall(text):
        if key.upper() == "QCD":
            max_qcd = int(value)
        else:
            max_qed = int(value)
    text = _ORDER_RE.sub(" ", text)

    body, _, veto = text.partition("/")
    statements = [s for s in body.replace("(", ",").replace(")", ",").split(",") if s.strip()]
    if not statements:
        raise ValueError("empty process")
    in_options, out_options = _split(statements[0], model)
    if not in_options or not out_options:
        raise ValueError("process needs initial and final state particles")
    decay_options = [_decay_options(s, model) for s in statements[1:]]
    excluded = frozenset(abs(p) for t in veto.split() for p in model.resolve(t))

    seen = set()
    processes = []
    for ins in product(*in_options):
        for outs in product(*out_options):
            if sum(model.charge3(p) for p in ins) != sum(model.charge3(p) for p in outs):
                continue
            for decays in product(*decay_options):
                top = _apply_decays(outs, decays, model)
                key = (tuple(sorted(ins)), tuple(sorted(s.canon() for s in top)))
                if key in seen:
                    continue
                seen.add(key)
                flat = [leaf for slot in top for leaf in slot.leaves()]
                if len(ins) + len(flat) < 3:
                    raise ValueError("process needs at least 3 external legs")
                position = {id(leaf): len(ins) + i for i, leaf in enumerate(flat)}
                resonances = tuple(
                    (slot.pdg, tuple(sorted(position[id(l)] for l in slot.leaves())))
                    for s in top
                    for slot in s.decayed()
                )
                processes.append(Process(
                    ins, tuple(leaf.pdg for leaf in flat), excluded, max_qcd, max_qed,
                    production=outs if decays else None, decays=decays, resonances=resonances, model=model,
                    loops=loops, one_pi=one_pi, tadpoles=tadpoles, external_corrections=external_corrections,
                ))
    return processes
