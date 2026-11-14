"""Corpora for CLOP2: a table with ``id``, ``text``, ``group`` and ``sequence`` (any lengths)."""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .model import PAD

_CODE = np.full(256, PAD, dtype=np.uint8)
_CODE[np.frombuffer(b"ACGTacgt", dtype=np.uint8)] = np.tile(np.arange(4, dtype=np.uint8), 2)
FIELD = re.compile(r"(\w+)=([^|]+)")


def encode_ragged(sequences) -> tuple[np.ndarray, np.ndarray]:
    """Strings of any length -> right-padded (n, max_len) uint8 codes and their lengths. Non-ACGT becomes PAD."""
    sequences = list(sequences)
    lengths = np.fromiter(map(len, sequences), dtype=np.int64, count=len(sequences))
    codes = np.full((len(sequences), lengths.max()), PAD, dtype=np.uint8)
    codes[np.arange(lengths.max())[None, :] < lengths[:, None]] = _CODE[
        np.frombuffer("".join(sequences).encode(), dtype=np.uint8)]
    return codes, lengths


def read_fasta(path: str) -> pd.DataFrame:
    """FASTA with CLOP headers ``>id|species=..|biotype=..|description=..|group=..``.

    The annotation text is biotype followed by description, lower-cased. ``group`` (for example a
    chromosome or gene family) keeps related sequences on one side of a split; it defaults to the id.
    """
    rows, opener = [], gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                head = line[1:].strip()
                fields = dict(FIELD.findall(head))
                identifier = head.split("|")[0].split()[0]
                text = " ".join(fields[k] for k in ("biotype", "description") if k in fields).lower()
                rows.append({"id": identifier, "text": text, "group": fields.get("group", identifier), "sequence": []})
            elif rows:
                rows[-1]["sequence"].append(line.strip())
    table = pd.DataFrame(rows)
    table["sequence"] = table["sequence"].map("".join)
    return table
