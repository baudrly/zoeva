"""Synthetic checks of the readers, splits, shuffles and metrics. Run: python -m pytest tests"""
import io
import struct

import numpy as np
import pandas as pd
import torch

from atlas.benchmark import Cells, confusion, macro_f1, vote_share
from atlas.common import HALF, canonical_map, dinucleotide_signature, encode, kmer_counts, revcomp, split_of
from atlas.features import covered, lift_offsets, merged
from atlas.genomes import genbank_features, twobit
from atlas.mechanisms import dinucleotide_shuffle, packed_kmers, permute_blocks
from clop2.model import Clop2, embed

RNG = np.random.default_rng(0)


def test_twobit_round_trip(tmp_path):
    seq = "ACGTTGCANNNNACGTA"
    packed = bytearray()
    order = {"T": 0, "C": 1, "A": 2, "G": 3, "N": 0}
    padded = seq + "T" * (-len(seq) % 4)
    for i in range(0, len(padded), 4):
        packed.append(sum(order[b] << (6 - 2 * j) for j, b in enumerate(padded[i:i + 4])))
    record = struct.pack("<4I", len(seq), 1, 8, 4) + struct.pack("<2I", 0, 0) + bytes(packed)
    name = b"chrT"
    header = struct.pack("<4I", 0x1A412743, 0, 1, 0) + bytes([len(name)]) + name
    offset = len(header) + 4
    (tmp_path / "t.2bit").write_bytes(header + struct.pack("<I", offset) + record)
    codes = twobit(tmp_path / "t.2bit", {"chrT"})["chrT"]
    assert "".join("ACGTN"[c] for c in codes) == seq


def test_genbank_features_parses_joins_and_strands():
    text = """FEATURES             Location/Qualifiers
     CDS             complement(join(10..20,30..40))
                     /locus_tag="b0001"
                     /note="first line
                     second line"
     tRNA            100..172
                     /product="tRNA-Ala"
ORIGIN
"""
    rows = list(genbank_features(io.StringIO(text)))
    assert rows[0][:3] == ("CDS", "-", [(9, 20), (29, 40)])
    assert rows[0][3]["note"] == ["first line second line"]
    assert rows[1][:3] == ("tRNA", "+", [(99, 172)])


def test_ecoli_split_drops_boundary_windows():
    centers = np.array([1_000_000, 2_000_000 + HALF, 2_000_000, 2_500_000, 3_000_000 - HALF, 3_200_000])
    out = split_of("ecoli", np.full(len(centers), "chr"), centers)
    assert list(out) == ["train", "test", "edge", "test", "edge", "validation"]


def test_dinucleotide_shuffle_keeps_counts_and_ends():
    row = RNG.integers(0, 4, 500).astype(np.uint8)
    for _ in range(20):
        shuffled = dinucleotide_shuffle(row, RNG)
        assert shuffled[0] == row[0] and shuffled[-1] == row[-1]
        assert np.array_equal(kmer_counts(shuffled[None], 2, canonical=False), kmer_counts(row[None], 2, canonical=False))


def test_block_permutation_keeps_blocks():
    codes = RNG.integers(0, 4, (3, 500)).astype(np.uint8)
    out = permute_blocks(codes, 10, RNG)
    assert sorted(map(bytes, out[0].reshape(-1, 10))) == sorted(map(bytes, codes[0].reshape(-1, 10)))


def test_strand_symmetric_features():
    codes = RNG.integers(0, 4, (5, 300)).astype(np.uint8)
    assert np.allclose(kmer_counts(codes, 4), kmer_counts(revcomp(codes), 4))
    assert np.allclose(dinucleotide_signature(codes), dinucleotide_signature(revcomp(codes)), atol=1e-6)
    assert np.array_equal(packed_kmers(codes, 32), packed_kmers(revcomp(codes), 32)[:, ::-1])
    assert canonical_map(4)[1] == 136


def test_kmer_counts_skip_n():
    codes = encode(["ACGTNACGT"])
    assert kmer_counts(codes, 4, canonical=False).sum() == 2


def test_lift_offsets_recovers_shift():
    genome = RNG.integers(0, 4, 200_000).astype(np.uint8)
    true_starts = np.sort(RNG.choice(np.arange(5_000, 190_000), 120, replace=False))
    shift = np.where(true_starts < 100_000, 0, -37)
    for p in true_starts:
        genome[p:p + 3] = (0, 3, 2)
    annotated = true_starts - shift
    starts = pd.DataFrame({"chrom": "c", "first": annotated, "strand": "+"})
    anchors, delta = lift_offsets(starts, {"c": genome})["c"]
    assert (delta == shift[np.argsort(annotated)]).mean() > 0.95


def test_interval_union_and_cover():
    union = merged(pd.DataFrame({"chrom": ["a", "a", "a"], "start": [10, 15, 50], "end": [20, 30, 60]}))
    assert union["a"][0].tolist() == [10, 50] and union["a"][1].tolist() == [30, 60]
    assert covered(union, np.array(["a"] * 4), np.array([9, 10, 29, 30])).tolist() == [False, True, True, False]


def test_metrics_and_votes():
    true = np.array([0, 0, 1, 1, 2])
    assert np.isclose(macro_f1(confusion(true, true, 4)), 1.0)
    votes = vote_share(np.array([[0, 1, 2]]), np.array([[0.9, 0.5, -0.2]]), np.array([1, 1, 0]), 2, 3)
    assert np.allclose(votes, [[0.0, 1.0]])
    cells = Cells(["human|CDS", "fly|CDS", "fly|tRNA"])
    assert cells.to_species.sum() == 3 and cells.names[0] == "fly|CDS"


def test_encoder_is_strand_and_padding_exact():
    torch.manual_seed(0)
    model = Clop2({"a": 0, "b": 1}).eval()
    codes = RNG.integers(0, 4, (4, 500)).astype(np.uint8)
    assert np.allclose(embed(model, codes), embed(model, revcomp(codes).copy()), atol=1e-5)
    padded = np.full((1, 500), 255, np.uint8)
    padded[0, :350] = codes[0, :350]
    assert np.allclose(embed(model, padded, np.array([350])), embed(model, codes[:1, :350]), atol=1e-5)


def test_hypothesis_helpers():
    from atlas.hypotheses import benjamini_hochberg, canonical_kmer, occurrences, similar
    from atlas.literature import aligned_similarity, consensus_matrix
    codes = encode(["AAATATAATCCC", "GGATTATAGGGG"])
    assert occurrences(codes, "TATAAT").sum(1).tolist() == [1, 1]
    assert similar("TATAAT", "ATTATA") and similar("TATAAT", "TATAAA") and not similar("TATAAT", "GCGCGC")
    assert np.allclose(benjamini_hochberg(np.array([0.01, 0.04, 0.03])), [0.03, 0.04, 0.04])
    assert canonical_kmer(0) == "AAAAAA"
    m = consensus_matrix("TATAAT")
    assert np.isclose(aligned_similarity(m, m[::-1, ::-1], True), 1.0)
    assert aligned_similarity(m, consensus_matrix("GCGCGC"), True) < 0.5


def test_motif_scan_covers_centre_on_both_strands():
    from atlas.asthma import covering_scores
    from atlas.common import HALF
    w = np.full((1, 4, 6), -1.0, np.float32)
    for j, b in enumerate([0, 0, 2, 3, 3, 1]):  # AAGTTC
        w[0, b, j] = 1.0
    weights, widths = torch.from_numpy(w), np.array([6])
    seq = np.full((1, 500), 1, np.uint8)  # all C: no match
    seq[0, HALF - 3:HALF + 3] = encode(["AAGTTC"])[0]
    assert covering_scores(seq, weights, widths)[0, 0] == 6
    rc = seq.copy()
    rc[0, HALF - 3:HALF + 3] = revcomp(encode(["AAGTTC"])[0])
    assert covering_scores(rc, weights, widths)[0, 0] == 6
    far = np.full((1, 500), 1, np.uint8)
    far[0, 10:16] = encode(["AAGTTC"])[0]
    assert covering_scores(far, weights, widths)[0, 0] < 6
