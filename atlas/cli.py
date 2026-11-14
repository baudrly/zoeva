"""Command line: python -m atlas.cli <command>."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read_json, write_json


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="atlas", description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--cache", default="genomes", help="directory of per-chromosome .npy files")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="build genomes cache, feature tables and manifest")
    prepare.add_argument("--sources", required=True, help="JSON file of input paths (see sources.example.json)")
    train = sub.add_parser("train", help="train one CLOP2 model and export embeddings")
    train.add_argument("--variant", default="main", help="main, compositional or without_<species>")
    train.add_argument("--seed-index", type=int, default=0)
    evaluate = sub.add_parser("evaluate", help="score all arms on validation or (once) test")
    evaluate.add_argument("--split", choices=("validation", "test"), required=True)
    sub.add_parser("freeze", help="write protocol.json from the validation run")
    sub.add_parser("transfer", help="compositional and leave-one-species-out analyses")
    sub.add_parser("mechanisms", help="shuffles, reading frame, landmarks, filters, geometry, loci")
    sub.add_parser("hypotheses", help="generate, register and test hypotheses; check them against the literature")
    explain = sub.add_parser("explain", help="test the replicated hypotheses of each FASTA record's top cell on that record")
    explain.add_argument("--fasta", required=True)
    explain.add_argument("--out", required=True, help="output JSON path")
    sub.add_parser("asthma", help="exploratory asthma variant mechanisms (Eliasen et al. 2022 starting point)")
    sub.add_parser("concept", help="data for the explanatory figures: label positions on the map and one worked hypothesis")
    sub.add_parser("figures", help="render figures from result files")
    sub.add_parser("export-web", help="build the browser app")
    hypothesize = sub.add_parser("hypothesize", help="rank species and sequence class for FASTA records")
    hypothesize.add_argument("--fasta", required=True)
    hypothesize.add_argument("--out", required=True, help="output prefix; writes .json and .md")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from .corpus import prepare as run
        report = run(read_json(args.sources), args.cache, args.results)
        print(json.dumps({"manifest_sha256": report["manifest_sha256"], "cells": len(report["cells"])}))
    elif args.command == "train":
        from .models import fit
        fit(str(Path(args.results) / "manifest.tsv"), str(Path(args.results) / "models"), args.variant, args.seed_index)
    elif args.command == "evaluate":
        from .benchmark import evaluate as run
        run(args.results, args.split)
    elif args.command == "freeze":
        from .benchmark import freeze
        freeze(args.results)
    elif args.command == "transfer":
        from .transfer import compositional, leave_one_species_out
        compositional(args.results)
        leave_one_species_out(args.results)
    elif args.command == "mechanisms":
        from .mechanisms import run
        run(args.results, args.cache)
    elif args.command == "hypotheses":
        from .hypotheses import run
        print(json.dumps(run(args.results), indent=1))
    elif args.command == "explain":
        from clop2.data import read_fasta
        from .hypotheses import explain
        from .pipeline import Zoeva
        zoeva, table = Zoeva(args.results), read_fasta(args.fasta)
        catalogue = read_json(Path(args.results) / "hypotheses" / "tested.json")["hypotheses"]
        write_json(args.out, {i: explain(zoeva, s, catalogue) for i, s in zip(table["id"], table["sequence"])})
    elif args.command == "asthma":
        from .asthma import run
        print(json.dumps(run(args.results, args.cache), indent=1))
    elif args.command == "concept":
        from .concept import run
        run(args.results)
    elif args.command == "figures":
        from .figures import render
        render(args.results)
    elif args.command == "export-web":
        from .export_web import build
        build(args.results, "webapp/template.html", "webapp/index.html")
    else:
        from clop2.data import read_fasta
        from .pipeline import Zoeva, markdown
        table = read_fasta(args.fasta)
        reports = Zoeva(args.results).hypothesize(dict(zip(table["id"], table["sequence"])))
        write_json(f"{args.out}.json", reports)
        Path(f"{args.out}.md").write_text(markdown(reports))


if __name__ == "__main__":
    main()
