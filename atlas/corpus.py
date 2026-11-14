"""The atlas corpus: one fixed 500 bp genomic window per annotated element, six species, held-out chromosomes.

Window length is the same for every class and species, so interval length carries no label information.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import genomes
from .common import (HALF, SEED, SPECIES, WINDOW, cell_text, decode, file_sha256, split_of, write_json)
from .features import BUILDERS, covered, is_start_codon, merged

CAPS = {"train": 1200, "validation": 200, "test": 400}
MINIMUM = {"train": 120, "validation": 10, "test": 25}
BUCKET = 100  # two windows of one class must sit in different 100 bp buckets
INTERGENIC = ("arabidopsis", "worm", "fly", "human")
INTERGENIC_MARGIN = 1000
INTERGENIC_DRAWS = 40_000
OFFSETS = np.arange(-HALF, HALF)


def intergenic(genome: dict, occupied: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Random positions at least INTERGENIC_MARGIN bases beyond the window edge from any gene or annotated element."""
    names = sorted(genome)
    sizes = np.array([len(genome[name]) for name in names])
    chrom = rng.choice(names, INTERGENIC_DRAWS, p=sizes / sizes.sum())
    center = (rng.random(INTERGENIC_DRAWS) * (sizes[np.searchsorted(names, chrom)] - WINDOW)).astype(np.int64) + HALF
    free = ~covered(merged(occupied, INTERGENIC_MARGIN + HALF), chrom, center)
    return pd.DataFrame({"chrom": chrom[free], "start": center[free], "end": center[free] + 1, "strand": "+",
                         "biotype": "intergenic", "group": "", "detail": ""})


def windows(genome: dict, chrom: np.ndarray, center: np.ndarray) -> np.ndarray:
    """(n, WINDOW) codes; rows running off a chromosome are filled with N."""
    out = np.full((len(center), WINDOW), 4, dtype=np.uint8)
    for name in np.unique(chrom):
        rows = np.flatnonzero((chrom == name) & (center >= HALF) & (center + HALF <= len(genome[name])))
        out[rows] = genome[name][center[rows, None] + OFFSETS]
    return out


def prepare(sources: dict, cache: str, out: str) -> dict:
    out_dir, rng, parts, report = Path(out), np.random.default_rng(SEED), [], {}
    (out_dir / "features").mkdir(parents=True, exist_ok=True)
    for species in SPECIES:
        if not (Path(cache) / species).exists():
            genomes.build(species, sources, cache)
        genome = genomes.load(species, cache)
        features, occupied, starts = BUILDERS[species](sources, genome)
        if species in INTERGENIC:
            features = pd.concat([features, intergenic(genome, occupied, rng)], ignore_index=True)
        features["center"] = (features["start"] + features["end"]) // 2
        features.to_csv(out_dir / "features" / f"{species}.tsv.gz", sep="\t", index=False)
        report[species] = {"chromosomes": {name: int(len(seq)) for name, seq in genome.items()},
                           "features": features["biotype"].value_counts().to_dict(), "dropped_cells": {}}
        if starts is not None:
            starts = starts[starts["chrom"].isin(genome)]
            report[species]["coding_starts_reading_ATG"] = float(is_start_codon(
                genome, starts["chrom"].to_numpy(), starts["first"].to_numpy(), starts["strand"].to_numpy()).mean())
        features["split"] = split_of(species, features["chrom"].to_numpy(), features["center"].to_numpy())
        features = features[features["split"] != "edge"].sample(frac=1, random_state=SEED)
        features = features[~features.assign(bucket=features["center"] // BUCKET).duplicated(["biotype", "chrom", "bucket"])]
        for biotype, cell in features.groupby("biotype"):
            chosen = []
            for split, cap in CAPS.items():
                rows = cell[cell["split"] == split].head(2 * cap + 50)
                codes = windows(genome, rows["chrom"].to_numpy(), rows["center"].to_numpy())
                keep = (codes < 4).all(1)
                chosen.append(rows[keep].head(cap).assign(sequence=decode(codes[keep][:cap]) if keep.any() else []))
            sizes = {split: len(rows) for split, rows in zip(CAPS, chosen)}
            if any(sizes[split] < MINIMUM[split] for split in CAPS):
                report[species]["dropped_cells"][biotype] = sizes
                continue
            parts.append(pd.concat(chosen).assign(species=species, text=cell_text(species, biotype)))
    manifest = pd.concat(parts, ignore_index=True)
    manifest["id"] = manifest["species"] + ":" + manifest["biotype"] + ":" + manifest["chrom"] + ":" + manifest["center"].astype(str)
    manifest["cell"] = manifest["species"] + "|" + manifest["biotype"]
    manifest = manifest[["id", "species", "biotype", "cell", "chrom", "center", "strand", "start", "end", "group",
                         "detail", "split", "text", "sequence"]]
    manifest.to_csv(out_dir / "manifest.tsv", sep="\t", index=False)
    report = {"window_bp": WINDOW, "seed": SEED, "caps": CAPS, "minimum_cell_size": MINIMUM, "species": report,
              "splits": {key: {"test": list(value[2]), "validation": list(value[3])} for key, value in SPECIES.items()},
              "cells": manifest.groupby(["cell", "split"]).size().unstack(fill_value=0).to_dict("index"),
              "manifest_sha256": file_sha256(out_dir / "manifest.tsv")}
    write_json(out_dir / "preparation.json", report)
    return report
