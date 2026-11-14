"""Curated sequence signals from the literature, and comparison of discovered frequency matrices with them and with JASPAR."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

LIBRARY_PATH = Path(__file__).resolve().parent.parent / "data" / "literature_signals.tsv"
IUPAC = {"A": "A", "C": "C", "G": "G", "T": "T", "R": "AG", "Y": "CT", "W": "AT", "S": "CG", "M": "AC", "K": "GT",
         "V": "ACG", "H": "ACT", "D": "AGT", "B": "CGT", "N": "ACGT"}
MATCH_MIN = 0.80
JASPAR_MIN = 0.90
MIN_OVERLAP = 6
TAXON = {"human": "vertebrates", "fly": "insects", "worm": "nematodes", "arabidopsis": "plants", "yeast": "fungi"}


def consensus_matrix(pattern: str) -> np.ndarray:
    return np.array([[1 / len(IUPAC[letter]) if base in IUPAC[letter] else 0 for base in "ACGT"] for letter in pattern])


@lru_cache(maxsize=1)
def library() -> pd.DataFrame:
    """One row per signal with its matrix, the species and classes where the literature expects it, and references."""
    table = pd.read_csv(LIBRARY_PATH, sep="\t")
    table["matrix"] = table["consensus"].map(consensus_matrix)
    for column in ("species", "classes", "references"):
        table[column] = table[column].str.split(",")
    return table


def aligned_similarity(query: np.ndarray, target: np.ndarray, centred: bool) -> float:
    """Best similarity of two (L, 4) matrices over all offsets with at least min(MIN_OVERLAP, shorter length)
    overlapping columns and over both strands of the target.

    centred=True: cosine of the background-subtracted (x - 0.25) overlapping columns (for consensus strings).
    centred=False: mean column-wise Pearson correlation (for count-based profiles such as JASPAR).
    """
    best = -1.0
    for strand in (target, target[::-1, ::-1]):
        need = min(MIN_OVERLAP, len(query), len(strand))
        for offset in range(-(len(strand) - need), len(query) - need + 1):
            q0, t0 = max(offset, 0), max(-offset, 0)
            width = min(len(query) - q0, len(strand) - t0)
            a, b = query[q0:q0 + width], strand[t0:t0 + width]
            if centred:
                a, b = a - 0.25, b - 0.25
                score = float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))
            else:
                a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
                score = float(np.mean((a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-9)))
            best = max(best, score)
    return best


def literature_match(pfm: np.ndarray, species: str, biotype: str) -> dict:
    """Closest curated signal, whether it clears MATCH_MIN, and whether the literature expects it in this cell."""
    table = library()
    scores = np.array([aligned_similarity(pfm, m, True) for m in table["matrix"]])
    i = int(scores.argmax())
    row = table.iloc[i]
    matched = scores[i] >= MATCH_MIN
    expected = matched and species in row["species"] and biotype in row["classes"]
    in_cell = np.array([species in r["species"] and biotype in r["classes"] for _, r in table.iterrows()])
    expected_any = bool((in_cell & (scores >= MATCH_MIN)).any())
    return {"name": row["name"] if matched else None, "closest": row["name"], "similarity": float(scores[i]),
            "consensus": row["consensus"], "expected_in_cell": bool(expected), "references": row["references"] if matched else [],
            "expected_context": f"{','.join(row['species'])}: {','.join(row['classes'])}", "note": row["note"],
            "expected_signal_above_threshold": expected_any,
            "expected_signals_above_threshold": table["name"][in_cell & (scores >= MATCH_MIN)].tolist()}


@lru_cache(maxsize=None)
def jaspar_profiles(taxon: str) -> tuple:
    """(name, ID, frequency matrix) of JASPAR CORE profiles of one taxonomic group (latest versions)."""
    from pyjaspar import jaspardb
    motifs = jaspardb().fetch_motifs(collection="CORE", tax_group=[taxon], all_versions=False)
    out = []
    for motif in motifs:
        counts = np.array([motif.counts[b] for b in "ACGT"], dtype=float).T
        out.append((motif.name, motif.matrix_id, counts / counts.sum(1, keepdims=True)))
    return tuple(out)


def jaspar_match(pfm: np.ndarray, species: str) -> dict | None:
    """Closest JASPAR CORE profile of the species' taxonomic group (annotation only; no taxon for E. coli)."""
    if species not in TAXON:
        return None
    profiles = jaspar_profiles(TAXON[species])
    scores = np.array([aligned_similarity(pfm, matrix, False) for _, _, matrix in profiles])
    i = int(scores.argmax())
    return {"name": profiles[i][0], "id": profiles[i][1], "similarity": float(scores[i]), "matched": bool(scores[i] >= JASPAR_MIN)}
