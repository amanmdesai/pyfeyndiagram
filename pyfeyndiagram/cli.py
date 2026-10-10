from __future__ import annotations

import argparse
import os
import re
import sys

from . import __version__
from .draw import write_pdf
from .generate import generate_all
from .model import LATEX, MULTIPARTICLES, NAMES, Model
from .tikz import tex_document


def _filename(process: str) -> str:
    name = process.replace("+", "p").replace("-", "m").replace("~", "bar").replace(">", " to ")
    name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return name or "process"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="pyfeyndiagram",
        description="Generate tree-level Standard Model Feynman diagrams as PDF and TikZ-Feynman LaTeX.",
    )
    parser.add_argument("process", nargs="?", help='MadGraph-style process, e.g. "e+ e- > mu+ mu-" or "g g > t t~ / h"')
    parser.add_argument("-o", "--outdir", default=".", help="output directory (default: .)")
    parser.add_argument("-n", "--name", help="output file stem (default: derived from the process)")
    parser.add_argument("--yukawa", default="t,b,ta", help="fermions with a Higgs coupling (default: t,b,ta)")
    parser.add_argument("--ckm", action="store_true", help="use full CKM mixing instead of diagonal")
    parser.add_argument("--max-qcd", type=int, help="maximum power of g_s")
    parser.add_argument("--max-qed", type=int, help="maximum power of e (electroweak and Higgs couplings)")
    parser.add_argument("--no-pdf", action="store_true", help="do not write the PDF")
    parser.add_argument("--no-tex", action="store_true", help="do not write the LaTeX file")
    parser.add_argument("--list-particles", action="store_true", help="list particle names and multiparticle labels, then exit")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    if args.list_particles:
        for pdg in sorted(NAMES, key=lambda p: (abs(p), -p)):
            print(f"{NAMES[pdg]:6s} pdg={pdg:<4d} ${LATEX[pdg]}$")
        for label, members in sorted(MULTIPARTICLES.items()):
            print(f"{label:6s} = {' '.join(members)}")
        return 0
    if not args.process:
        parser.error("a process is required")

    try:
        model = Model(yukawa=[s for s in args.yukawa.split(",") if s], ckm=args.ckm)
        results = generate_all(args.process, model, args.max_qcd, args.max_qed)
    except (ValueError, KeyError) as err:
        parser.error(str(err))

    if not results:
        print("no tree-level diagrams for this process", file=sys.stderr)
        return 1

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.join(args.outdir, args.name or _filename(args.process))
    written = []
    if not args.no_pdf:
        write_pdf(results, stem + ".pdf")
        written.append(stem + ".pdf")
    if not args.no_tex:
        with open(stem + ".tex", "w") as handle:
            handle.write(tex_document(results))
        written.append(stem + ".tex")

    for process, diagrams in results:
        print(f"{process}: {len(diagrams)} diagrams")
    if written:
        print("wrote " + " and ".join(written))
    return 0
