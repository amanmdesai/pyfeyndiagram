from __future__ import annotations

import argparse
import os
import re
import sys

from .draw import write_pdf
from .generate import generate_all
from .model import Model
from .tikz import tex_document


def _filename(process: str) -> str:
    name = process.replace("+", "p").replace("-", "m").replace("~", "bar").replace(">", " to ")
    name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return name or "process"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="feyndiag",
        description="Generate tree-level Standard Model Feynman diagrams as PDF and TikZ-Feynman LaTeX.",
    )
    parser.add_argument("process", help='MadGraph-style process, e.g. "e+ e- > mu+ mu-" or "g g > t t~ / h"')
    parser.add_argument("-o", "--outdir", default=".", help="output directory (default: .)")
    parser.add_argument("-n", "--name", help="output file stem (default: derived from the process)")
    parser.add_argument("--yukawa", default="t,b,ta", help="fermions with a Higgs coupling (default: t,b,ta)")
    parser.add_argument("--ckm", action="store_true", help="use full CKM mixing instead of diagonal")
    parser.add_argument("--max-qcd", type=int, help="maximum power of g_s")
    parser.add_argument("--max-qed", type=int, help="maximum power of e (electroweak and Higgs couplings)")
    args = parser.parse_args(argv)

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
    write_pdf(results, stem + ".pdf")
    with open(stem + ".tex", "w") as handle:
        handle.write(tex_document(results))

    for process, diagrams in results:
        print(f"{process}: {len(diagrams)} diagrams")
    print(f"wrote {stem}.pdf and {stem}.tex")
    return 0
