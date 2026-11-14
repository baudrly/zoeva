"""Species and sequence-class registries, window geometry, encoding and k-mer features."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

SEED = 20261114
WINDOW = 500
HALF = WINDOW // 2
CROPS = (200, 350, 500)
N = 4  # code of any base that is not A, C, G or T

# key -> (binomial, lineage tokens for the text encoder, test chromosomes, validation chromosomes)
SPECIES = {
    "ecoli": ("Escherichia coli K-12", "escherichia coli bacterium prokaryote", (), ()),
    "yeast": ("Saccharomyces cerevisiae", "saccharomyces cerevisiae yeast fungus eukaryote", ("chr04", "chr15"), ("chr07",)),
    "arabidopsis": ("Arabidopsis thaliana", "arabidopsis thaliana plant eukaryote", ("Chr4",), ("Chr2",)),
    "worm": ("Caenorhabditis elegans", "caenorhabditis elegans nematode invertebrate animal eukaryote", ("III",), ("V",)),
    "fly": ("Drosophila melanogaster", "drosophila melanogaster insect invertebrate animal eukaryote", ("2R",), ("X",)),
    "human": ("Homo sapiens", "homo sapiens mammal vertebrate animal eukaryote", ("chr2", "chr11"), ("chr12",)),
}
SPECIES_KEYS = tuple(SPECIES)
# The single E. coli chromosome is split by coordinate instead of by chromosome.
ECOLI_TEST, ECOLI_VALIDATION = (2_000_000, 3_000_000), (3_000_000, 3_500_000)

# sequence class -> (Sequence Ontology style name, tokens for the text encoder)
BIOTYPES = {
    "CDS": ("coding sequence", "coding sequence cds protein exon genic transcribed"),
    "five_prime_UTR": ("5' UTR", "five prime utr untranslated exon genic transcribed"),
    "three_prime_UTR": ("3' UTR", "three prime utr untranslated exon genic transcribed"),
    "intron": ("intron", "intron spliced genic transcribed noncoding"),
    "intergenic": ("intergenic", "intergenic region unannotated noncoding"),
    "promoter": ("promoter", "promoter transcription start site regulatory"),
    "enhancer": ("enhancer", "enhancer distal regulatory"),
    "open_chromatin": ("open chromatin", "open chromatin accessible regulatory"),
    "lncRNA": ("lncRNA", "long noncoding rna lncrna gene transcribed"),
    "pseudogene": ("pseudogene", "pseudogene decayed gene copy"),
    "tRNA": ("tRNA", "transfer rna trna gene small structured"),
    "snRNA": ("snRNA", "small nuclear rna snrna gene small structured"),
    "snoRNA": ("snoRNA", "small nucleolar rna snorna gene small structured"),
    "miRNA": ("miRNA", "microrna mirna hairpin precursor gene small"),
    "transposable_element": ("transposable element", "transposable element repeat mobile"),
    "replication_origin": ("replication origin", "replication origin autonomously replicating"),
}
BIOTYPE_KEYS = tuple(BIOTYPES)


def cell_text(species: str, biotype: str) -> str:
    """Annotation text of one species x sequence-class cell: lineage tokens followed by class tokens."""
    return f"{SPECIES[species][1]} {BIOTYPES[biotype][1]}"


def split_of(species: str, chrom: np.ndarray, center: np.ndarray) -> np.ndarray:
    """train / validation / test per window; whole chromosomes, or coordinate blocks for E. coli."""
    out = np.full(len(center), "train", dtype=object)
    if species == "ecoli":
        for name, (lo, hi) in (("test", ECOLI_TEST), ("validation", ECOLI_VALIDATION)):
            out[(center >= lo - HALF) & (center < hi + HALF)] = "edge"  # windows straddling a block boundary
            out[(center >= lo + HALF) & (center < hi - HALF)] = name
        return out
    _, _, test, validation = SPECIES[species]
    out[np.isin(chrom, test)] = "test"
    out[np.isin(chrom, validation)] = "validation"
    return out


_CODE = np.full(256, N, dtype=np.uint8)
_CODE[np.frombuffer(b"ACGTacgt", dtype=np.uint8)] = np.tile(np.arange(4, dtype=np.uint8), 2)
LETTERS = np.frombuffer(b"ACGTN", dtype=np.uint8)


def encode_bytes(raw: bytes) -> np.ndarray:
    return _CODE[np.frombuffer(raw, dtype=np.uint8)]


def encode(seqs) -> np.ndarray:
    """Equal-length strings -> (n, L) uint8 codes (A0 C1 G2 T3, anything else 4)."""
    seqs = list(seqs)
    return encode_bytes("".join(seqs).encode()).reshape(len(seqs), -1)


def decode(codes: np.ndarray) -> list[str]:
    return [row.tobytes().decode() for row in LETTERS[np.atleast_2d(codes)]]


def revcomp(codes: np.ndarray) -> np.ndarray:
    """Reverse complement along the last axis; N stays N."""
    return np.where(codes < 4, 3 - codes, codes)[..., ::-1]


def crop(codes: np.ndarray, length: int) -> np.ndarray:
    start = (codes.shape[1] - length) // 2
    return codes[:, start:start + length]


@lru_cache(maxsize=None)
def canonical_map(k: int) -> tuple[np.ndarray, int]:
    """Map every k-mer index to a dense id shared with its reverse complement."""
    idx = np.arange(4**k, dtype=np.int64)
    rc, rest = np.zeros_like(idx), idx.copy()
    for _ in range(k):
        rc = (rc << 2) | (3 - (rest & 3))
        rest >>= 2
    uniq, dense = np.unique(np.minimum(idx, rc), return_inverse=True)
    return dense.astype(np.int32), len(uniq)


def kmer_counts(codes: np.ndarray, k: int, canonical: bool = True, chunk: int = 4096) -> np.ndarray:
    """k-mer counts per row, float32. Canonical profiles merge a k-mer with its reverse complement.
    k-mers touching a non-ACGT base are not counted."""
    dense, size = canonical_map(k) if canonical else (np.arange(4**k, dtype=np.int32), 4**k)
    width = codes.shape[1] - k + 1
    out = np.empty((len(codes), size), dtype=np.float32)
    for lo in range(0, len(codes), chunk):
        part = codes[lo:lo + chunk]
        raw, valid = np.zeros((len(part), width), dtype=np.int64), np.ones((len(part), width), dtype=bool)
        for j in range(k):
            column = part[:, j:j + width]
            raw, valid = (raw << 2) | np.minimum(column, 3), valid & (column < 4)
        rows = np.arange(len(part), dtype=np.int64)[:, None] * size
        out[lo:lo + chunk] = np.bincount((rows + dense[raw]).ravel(), valid.ravel(), len(part) * size).reshape(len(part), size)
    return out


def kmer_profile(codes: np.ndarray, k: int) -> np.ndarray:
    """L2-normalised canonical k-mer profile."""
    out = kmer_counts(codes, k)
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
    return out


def dinucleotide_signature(codes: np.ndarray) -> np.ndarray:
    """Strand-symmetrised dinucleotide odds ratios rho*_XY = f_XY / (f_X f_Y) (Karlin & Burge 1995), 16 columns,
    followed by GC fraction."""
    both = np.concatenate([codes, np.full((len(codes), 1), N, np.uint8), revcomp(codes)], axis=1)
    mono = np.stack([(both == b).sum(1) for b in range(4)], 1).astype(np.float64)
    mono /= np.maximum(mono.sum(1, keepdims=True), 1)
    di = kmer_counts(both, 2, canonical=False).astype(np.float64)
    di /= np.maximum(di.sum(1, keepdims=True), 1)
    expected = (mono[:, :, None] * mono[:, None, :]).reshape(len(codes), 16)
    return np.column_stack([di / np.maximum(expected, 1e-9), mono[:, 1] + mono[:, 2]]).astype(np.float32)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def read_json(path: str | Path):
    return json.loads(Path(path).read_text())
