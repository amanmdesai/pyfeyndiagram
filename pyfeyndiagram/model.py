from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

KINDS = ("fermion", "photon", "boson", "gluon", "scalar", "ghost")

UP_QUARKS = (2, 4, 6)
DOWN_QUARKS = (1, 3, 5)
CHARGED_LEPTONS = (11, 13, 15)
NEUTRINOS = (12, 14, 16)
FERMIONS = frozenset(UP_QUARKS + DOWN_QUARKS + CHARGED_LEPTONS + NEUTRINOS)

_SM_PARTICLES = [
    (1, "d", "d", "fermion", -1 / 3, False),
    (2, "u", "u", "fermion", 2 / 3, False),
    (3, "s", "s", "fermion", -1 / 3, False),
    (4, "c", "c", "fermion", 2 / 3, False),
    (5, "b", "b", "fermion", -1 / 3, False),
    (6, "t", "t", "fermion", 2 / 3, False),
    (11, "e-", "e^{-}", "fermion", -1, False),
    (12, "ve", r"\nu_{e}", "fermion", 0, False),
    (13, "mu-", r"\mu^{-}", "fermion", -1, False),
    (14, "vm", r"\nu_{\mu}", "fermion", 0, False),
    (15, "ta-", r"\tau^{-}", "fermion", -1, False),
    (16, "vt", r"\nu_{\tau}", "fermion", 0, False),
    (21, "g", "g", "gluon", 0, True),
    (22, "a", r"\gamma", "photon", 0, True),
    (23, "z", "Z", "boson", 0, True),
    (24, "w+", "W^{+}", "boson", 1, False),
    (25, "h", "H", "scalar", 0, True),
    (82, "ghg", "c", "ghost", 0, False),
]

_SM_ALIASES = {"gamma": "a", "tau-": "ta-", "tau+": "ta+"}

_SM_MULTIPARTICLES = {
    "j": ["g", "u", "d", "c", "s", "u~", "d~", "c~", "s~"],
    "p": ["g", "u", "d", "c", "s", "u~", "d~", "c~", "s~"],
    "l+": ["e+", "mu+", "ta+"],
    "l-": ["e-", "mu-", "ta-"],
    "vl": ["ve", "vm", "vt"],
    "vl~": ["ve~", "vm~", "vt~"],
}

_YUKAWA_FLAVOURS = {
    "e": 11, "mu": 13, "ta": 15, "u": 2, "d": 1, "s": 3, "c": 4, "b": 5, "t": 6,
}


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


@dataclass(frozen=True)
class Particle:
    """A particle with positive PDG id; its antiparticle (if any) is ``-pdg``."""

    pdg: int
    name: str
    latex: str
    kind: str
    charge3: int
    self_conjugate: bool
    anti_name: str
    anti_latex: str


class Model:
    """Tree-level vertices in all-incoming convention; the SM in unitary gauge by default.

    Orders are (QCD, QED) powers of the couplings g_s and e; electroweak and
    Higgs couplings all count towards QED. Extend the model with
    :meth:`add_particle`, :meth:`add_vertex` and :meth:`add_multiparticle`.
    """

    def __init__(self, yukawa=("t", "b", "ta"), ckm: bool = False):
        self.yukawa = tuple(yukawa)
        self.ckm = ckm
        self.particles: dict[int, Particle] = {}
        self.multiparticles: dict[str, list[str]] = {}
        self.vertices: dict[tuple[int, ...], tuple[int, int]] = {}
        self._lookup: dict[str, int] = {}
        self._completions = None
        self._load_sm()

    def add_particle(
        self, pdg: int, name: str, latex: str, kind: str, charge: float = 0,
        self_conjugate: bool = False, anti_name: str = None, anti_latex: str = None,
    ) -> Particle:
        """Define a particle with positive ``pdg``; ``-pdg`` is its antiparticle.

        ``kind`` is the line style: fermion, photon, boson, gluon, scalar or ghost.
        ``charge`` is in units of e. The antiparticle name and LaTeX are derived
        (``x`` -> ``x~``, ``x+`` -> ``x-``) unless given.
        """
        if pdg <= 0:
            raise ValueError("pdg must be positive; the antiparticle is -pdg")
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {', '.join(KINDS)}, not {kind!r}")
        if self_conjugate and kind in ("fermion", "ghost"):
            raise ValueError("self-conjugate (Majorana) fermions and ghosts are not supported")
        if self_conjugate and charge:
            raise ValueError("a self-conjugate particle must be neutral")
        particle = Particle(
            pdg, name, latex, kind, round(3 * charge), self_conjugate,
            name if self_conjugate else anti_name or _anti_name(name),
            latex if self_conjugate else anti_latex or _anti_latex(latex),
        )
        names = {particle.name.lower(), particle.anti_name.lower()}
        if pdg in self.particles or any(n in self._lookup for n in names):
            raise ValueError(f"particle {name!r} (pdg {pdg}) is already defined")
        self.particles[pdg] = particle
        self._lookup[particle.name.lower()] = pdg
        self._lookup[particle.anti_name.lower()] = pdg if self_conjugate else -pdg
        return particle

    def add_alias(self, alias: str, name: str) -> None:
        self._lookup[alias.lower()] = self.pdg(name)

    def add_multiparticle(self, label: str, names) -> None:
        names = list(names)
        for name in names:
            self.pdg(name)
        self.multiparticles[label.lower()] = names

    def add_vertex(self, qcd: int, qed: int, *fields) -> None:
        """Add a 3- or 4-point vertex of coupling orders (qcd, qed).

        Fields are particle names or signed PDG ids, all-incoming: ``("e-", "e+", "zp")``
        is the vertex where an incoming e-, an incoming e+ and an incoming Z' meet.
        """
        labels = tuple(self._label(f) for f in fields)
        if len(labels) not in (3, 4):
            raise ValueError("a vertex needs 3 or 4 fields")
        if sum(self.charge3(p) for p in labels):
            raise ValueError(f"vertex {fields} does not conserve charge")
        self.vertices[tuple(sorted(labels))] = (qcd, qed)
        self._completions = None

    @property
    def completions(self):
        if self._completions is None:
            completions = defaultdict(list)
            for key, (qcd, qed) in self.vertices.items():
                for missing in set(key):
                    rest = list(key)
                    rest.remove(missing)
                    completions[tuple(rest)].append((missing, qcd, qed))
            self._completions = completions
        return self._completions

    def pdg(self, name: str) -> int:
        try:
            return self._lookup[name.lower()]
        except KeyError:
            raise ValueError(f"unknown particle {name!r}") from None

    def resolve(self, token: str) -> list:
        """PDG ids for a particle name or multiparticle label."""
        members = self.multiparticles.get(token.lower())
        return [self.pdg(n) for n in members] if members is not None else [self.pdg(token)]

    def pdgs(self) -> list:
        """All signed PDG ids, ordered by |pdg| with the particle first."""
        found = []
        for pdg in sorted(self.particles):
            found.append(pdg)
            if not self.particles[pdg].self_conjugate:
                found.append(-pdg)
        return found

    def name(self, pdg: int) -> str:
        p = self.particles[abs(pdg)]
        return p.name if pdg > 0 else p.anti_name

    def latex(self, pdg: int) -> str:
        p = self.particles[abs(pdg)]
        return p.latex if pdg > 0 else p.anti_latex

    def charge3(self, pdg: int) -> int:
        c = self.particles[abs(pdg)].charge3
        return c if pdg > 0 else -c

    def kind(self, pdg: int) -> str:
        return self.particles[abs(pdg)].kind

    def has_arrow(self, pdg: int) -> bool:
        return self.particles[abs(pdg)].kind in ("fermion", "ghost")

    def anti(self, pdg: int) -> int:
        return pdg if self.particles[abs(pdg)].self_conjugate else -pdg

    def _label(self, field) -> int:
        if isinstance(field, str):
            return self.pdg(field)
        particle = self.particles.get(abs(field))
        if particle is None:
            raise ValueError(f"unknown particle {field!r}")
        return abs(field) if particle.self_conjugate else field

    def _load_sm(self) -> None:
        for pdg, name, latex, kind, charge, self_conjugate in _SM_PARTICLES:
            self.add_particle(pdg, name, latex, kind, charge, self_conjugate)
        for alias, name in _SM_ALIASES.items():
            self.add_alias(alias, name)
        for label, names in _SM_MULTIPARTICLES.items():
            self.add_multiparticle(label, names)

        add = self.add_vertex
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

        add(1, 0, 82, -82, 21)
