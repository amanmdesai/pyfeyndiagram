from __future__ import annotations

from collections import defaultdict

SELF_CONJUGATE = frozenset({21, 22, 23, 25})
FERMIONS = frozenset({1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16})
UP_QUARKS = (2, 4, 6)
DOWN_QUARKS = (1, 3, 5)
CHARGED_LEPTONS = (11, 13, 15)
NEUTRINOS = (12, 14, 16)

_BASE_NAMES = {
    1: "d", 2: "u", 3: "s", 4: "c", 5: "b", 6: "t",
    11: "e-", 12: "ve", 13: "mu-", 14: "vm", 15: "ta-", 16: "vt",
    21: "g", 22: "a", 23: "z", 24: "w+", 25: "h",
}

_BASE_LATEX = {
    1: "d", 2: "u", 3: "s", 4: "c", 5: "b", 6: "t",
    11: "e^{-}", 12: r"\nu_{e}", 13: r"\mu^{-}", 14: r"\nu_{\mu}",
    15: r"\tau^{-}", 16: r"\nu_{\tau}",
    21: "g", 22: r"\gamma", 23: "Z", 24: "W^{+}", 25: "H",
}

_CHARGE3 = {1: -1, 2: 2, 3: -1, 4: 2, 5: -1, 6: 2, 11: -3, 13: -3, 15: -3, 24: 3}

_YUKAWA_FLAVOURS = {
    "e": 11, "mu": 13, "ta": 15, "u": 2, "d": 1, "s": 3, "c": 4, "b": 5, "t": 6,
}

_ALIASES = {"gamma": "a", "tau-": "ta-", "tau+": "ta+"}

MULTIPARTICLES = {
    "j": ["g", "u", "d", "c", "s", "u~", "d~", "c~", "s~"],
    "l+": ["e+", "mu+", "ta+"],
    "l-": ["e-", "mu-", "ta-"],
    "vl": ["ve", "vm", "vt"],
    "vl~": ["ve~", "vm~", "vt~"],
}
MULTIPARTICLES["p"] = MULTIPARTICLES["j"]


def _anti_name(name: str) -> str:
    if name.endswith("-"):
        return name[:-1] + "+"
    if name.endswith("+"):
        return name[:-1] + "-"
    return name + "~"


def _anti_latex(tex: str) -> str:
    if tex.endswith("^{-}"):
        return tex[:-4] + "^{+}"
    if tex.endswith("^{+}"):
        return tex[:-4] + "^{-}"
    if "_" in tex:
        head, sub = tex.split("_", 1)
        return rf"\bar{{{head}}}_{sub}"
    return rf"\bar{{{tex}}}"


NAMES: dict[int, str] = {}
LATEX: dict[int, str] = {}
for _pdg, _name in _BASE_NAMES.items():
    NAMES[_pdg] = _name
    LATEX[_pdg] = _BASE_LATEX[_pdg]
    if _pdg not in SELF_CONJUGATE:
        NAMES[-_pdg] = _anti_name(_name)
        LATEX[-_pdg] = _anti_latex(_BASE_LATEX[_pdg])

PDG_BY_NAME: dict[str, int] = {name: pdg for pdg, name in NAMES.items()}
for _alias, _target in _ALIASES.items():
    PDG_BY_NAME[_alias] = PDG_BY_NAME[_target]


def anti(pdg: int) -> int:
    return pdg if abs(pdg) in SELF_CONJUGATE else -pdg


def charge3(pdg: int) -> int:
    c = _CHARGE3.get(abs(pdg), 0)
    return c if pdg > 0 else -c


def kind(pdg: int) -> str:
    a = abs(pdg)
    if a in FERMIONS:
        return "fermion"
    return {21: "gluon", 22: "photon", 23: "boson", 24: "boson", 25: "scalar"}[a]


class Model:
    """Tree-level SM vertices in unitary gauge, all-incoming convention.

    Orders are (QCD, QED) powers of the couplings g_s and e; electroweak and
    Higgs couplings all count towards QED.
    """

    def __init__(self, yukawa=("t", "b", "ta"), ckm: bool = False):
        self.yukawa = tuple(yukawa)
        self.ckm = ckm
        self.vertices = self._build_vertices()
        self.completions = self._build_completions()

    def _build_vertices(self) -> dict[tuple[int, ...], tuple[int, int]]:
        vertices: dict[tuple[int, ...], tuple[int, int]] = {}

        def add(qcd, qed, *labels):
            vertices[tuple(sorted(labels))] = (qcd, qed)

        quarks = UP_QUARKS + DOWN_QUARKS
        for q in quarks:
            add(1, 0, q, -q, 21)
        for f in quarks + CHARGED_LEPTONS:
            add(0, 1, f, -f, 22)
        for f in FERMIONS:
            add(0, 1, f, -f, 23)
        for name in self.yukawa:
            f = _YUKAWA_FLAVOURS[name]
            add(0, 1, f, -f, 25)
        for lep, nu in zip(CHARGED_LEPTONS, NEUTRINOS):
            add(0, 1, lep, -nu, 24)
            add(0, 1, -lep, nu, -24)
        if self.ckm:
            pairs = [(u, d) for u in UP_QUARKS for d in DOWN_QUARKS]
        else:
            pairs = list(zip(UP_QUARKS, DOWN_QUARKS))
        for u, d in pairs:
            add(0, 1, u, -d, -24)
            add(0, 1, -u, d, 24)

        add(1, 0, 21, 21, 21)
        add(2, 0, 21, 21, 21, 21)
        add(0, 1, 24, -24, 22)
        add(0, 1, 24, -24, 23)
        add(0, 2, 24, 24, -24, -24)
        add(0, 2, 24, -24, 23, 23)
        add(0, 2, 24, -24, 22, 22)
        add(0, 2, 24, -24, 22, 23)

        add(0, 1, 25, 24, -24)
        add(0, 1, 25, 23, 23)
        add(0, 1, 25, 25, 25)
        add(0, 2, 25, 25, 24, -24)
        add(0, 2, 25, 25, 23, 23)
        add(0, 2, 25, 25, 25, 25)
        return vertices

    def _build_completions(self):
        completions = defaultdict(list)
        for key, (qcd, qed) in self.vertices.items():
            for missing in set(key):
                rest = list(key)
                rest.remove(missing)
                completions[tuple(rest)].append((missing, qcd, qed))
        return completions
