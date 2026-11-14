"""CLOP2 training runs on the atlas corpus and their embedding exports."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clop2.model import embed, load
from clop2.train import train

from .common import CROPS, SEED, SPECIES, encode, write_json

EPOCHS = {"main": 10, "compositional": 10}  # leave-one-species-out runs use LOSO_EPOCHS
LOSO_EPOCHS = 6
# Cells withheld from the "compositional" model: its species and its sequence class are both seen in training,
# their combination never is.
HELD_OUT_CELLS = ("ecoli|CDS", "yeast|tRNA", "arabidopsis|intron", "worm|three_prime_UTR", "fly|promoter",
                  "fly|enhancer", "human|five_prime_UTR", "human|pseudogene")


def variant_rows(manifest: pd.DataFrame, variant: str) -> pd.Series:
    """Training rows of a variant: ``main``, ``compositional`` or ``without_<species>``."""
    train_rows = manifest["split"] == "train"
    if variant == "main":
        return train_rows
    if variant == "compositional":
        return train_rows & ~manifest["cell"].isin(HELD_OUT_CELLS)
    species = variant.removeprefix("without_")
    if species not in SPECIES:
        raise ValueError(f"unknown variant {variant}")
    return train_rows & (manifest["species"] != species)


def run_name(variant: str, seed_index: int) -> str:
    return f"clop2_{variant}_s{seed_index}"


def fit(manifest_path: str, out: str, variant: str, seed_index: int) -> Path:
    """Train one model and export unit-norm embeddings of every manifest row."""
    torch.set_num_threads(1)
    manifest = pd.read_csv(manifest_path, sep="\t", keep_default_na=False)
    name = run_name(variant, seed_index)
    checkpoint = train(manifest[variant_rows(manifest, variant)], out, name, SEED + seed_index, CROPS,
                       epochs=EPOCHS.get(variant, LOSO_EPOCHS))
    model, state = load(checkpoint)
    np.savez_compressed(Path(out) / f"{name}.embeddings.npz", ids=manifest["id"].to_numpy(str),
                        dna=embed(model, encode(manifest["sequence"])).astype(np.float32))
    write_json(Path(out) / f"{name}.provenance.json", {k: v for k, v in state.items() if k not in ("model", "vocab", "texts")}
               | {"variant": variant, "training_cells": sorted(manifest.loc[variant_rows(manifest, variant), "cell"].unique())})
    return checkpoint
