from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import product
from typing import Optional

from .model import LATEX, MULTIPARTICLES, NAMES, PDG_BY_NAME, anti, charge3

_ORDER_RE = re.compile(r"\b(QCD|QED)\s*(?:<=|=)\s*(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class Process:
    incoming: tuple
    outgoing: tuple
    excluded: frozenset = frozenset()
    max_qcd: Optional[int] = None
    max_qed: Optional[int] = None

    @property
    def legs(self) -> tuple:
        """All external labels in the all-incoming convention."""
        return tuple(self.incoming) + tuple(anti(p) for p in self.outgoing)

    def __str__(self) -> str:
        text = " ".join(NAMES[p] for p in self.incoming)
        text += " > " + " ".join(NAMES[p] for p in self.outgoing)
        if self.excluded:
            text += " / " + " ".join(NAMES[p] for p in sorted(self.excluded))
        return text

    def latex(self) -> str:
        ins = r"\,".join(LATEX[p] for p in self.incoming)
        outs = r"\,".join(LATEX[p] for p in self.outgoing)
        return rf"{ins} \to {outs}"


def _resolve(token: str) -> list:
    key = token.lower()
    if key in MULTIPARTICLES:
        return [PDG_BY_NAME[n] for n in MULTIPARTICLES[key]]
    if key in PDG_BY_NAME:
        return [PDG_BY_NAME[key]]
    raise ValueError(f"unknown particle {token!r}")


def parse_process(text: str, max_qcd=None, max_qed=None) -> list:
    """Parse a MadGraph-style process string into explicit subprocesses.

    Syntax: ``e+ e- > mu+ mu- [/ z h] [QCD=n] [QED=n]``. Multiparticle labels
    (``j``, ``p``, ``l+``, ``l-``, ``vl``, ``vl~``) are expanded.
    """
    for key, value in _ORDER_RE.findall(text):
        if key.upper() == "QCD":
            max_qcd = int(value)
        else:
            max_qed = int(value)
    text = _ORDER_RE.sub(" ", text)

    body, _, veto = text.partition("/")
    if body.count(">") != 1:
        raise ValueError("process must contain exactly one '>'")
    lhs, rhs = body.split(">")
    in_options = [_resolve(t) for t in lhs.split()]
    out_options = [_resolve(t) for t in rhs.split()]
    if not in_options or not out_options:
        raise ValueError("process needs initial and final state particles")
    if len(in_options) + len(out_options) < 3:
        raise ValueError("process needs at least 3 external legs")
    excluded = frozenset(abs(p) for t in veto.split() for p in _resolve(t))

    seen = set()
    processes = []
    for ins in product(*in_options):
        for outs in product(*out_options):
            if sum(charge3(p) for p in ins) != sum(charge3(p) for p in outs):
                continue
            key = (tuple(sorted(ins)), tuple(sorted(outs)))
            if key in seen:
                continue
            seen.add(key)
            processes.append(Process(ins, outs, excluded, max_qcd, max_qed))
    return processes
