"""Per-species feature tables: where each sequence class sits in each genome.

Every builder returns (features, occupied, coding starts). ``features`` has one row per annotated element
with 0-based half-open coordinates; ``occupied`` lists the intervals that an "intergenic" window must avoid;
coding starts (None where the source has none) feed the start-codon check of the coordinate system.
"""
from __future__ import annotations

import gzip
import io
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from .genomes import genbank_features, yeast_records

COLUMNS = ["chrom", "start", "end", "strand", "biotype", "group", "detail"]
MIN_SEGMENT, MIN_INTRON, MIN_REPEAT = 100, 200, 200
HUMAN_RNA = {"lincRNA": "lncRNA", "snRNA": "snRNA", "snoRNA": "snoRNA", "miRNA": "miRNA",
             "processed_pseudogene": "pseudogene"}
FLY_CHROMS = ("2L", "2R", "3L", "3R", "4", "X")
FLY_SMALL = {"tRNA": "tRNA", "snoRNA": "snoRNA", "pre_miRNA": "miRNA"}
FLY_EXONIC = {"ncRNA": "lncRNA", "pseudogene": "pseudogene"}
LIFT_RANGE, LIFT_SMOOTH, LIFT_REFINE = 600, 21, 4


def table(chrom, start, end, strand, biotype, group, detail="") -> pd.DataFrame:
    return pd.DataFrame({"chrom": chrom, "start": start, "end": end, "strand": strand, "biotype": biotype,
                         "group": group, "detail": detail})[COLUMNS]


def bed(path: Path, strip_chr: bool = False) -> pd.DataFrame:
    out = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3, 5], names=["chrom", "start", "end", "group", "strand"],
                      dtype={"chrom": str})
    if strip_chr:
        out["chrom"] = out["chrom"].str.removeprefix("chr")
    return out


def segments(rows: pd.DataFrame, biotype: str, min_len: int) -> pd.DataFrame:
    rows = rows[rows["end"] - rows["start"] >= min_len]
    return table(rows["chrom"], rows["start"], rows["end"], rows["strand"], biotype, rows["group"], rows.get("detail", ""))


def tss(genes: pd.DataFrame, offset: int = 0) -> pd.DataFrame:
    """One-base promoter anchors: the 5' end of each gene, moved ``offset`` bases upstream."""
    plus = (genes["strand"] == "+").to_numpy()
    site = np.where(plus, genes["start"] - offset, genes["end"] - 1 + offset)
    return table(genes["chrom"], site, site + 1, genes["strand"], "promoter", genes["group"])


def merged(intervals: pd.DataFrame, margin: int = 0) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Union of intervals per chromosome as sorted, disjoint (starts, ends), each widened by ``margin``."""
    out = {}
    for chrom, part in intervals.groupby("chrom"):
        order = np.argsort(part["start"].to_numpy(), kind="stable")
        start, end = part["start"].to_numpy()[order] - margin, np.maximum.accumulate(part["end"].to_numpy()[order] + margin)
        first = np.r_[True, start[1:] > end[:-1]]
        out[chrom] = (start[first], end[np.r_[first[1:], True]])
    return out


def covered(union: dict, chrom: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Whether each position falls inside the union of intervals on its chromosome."""
    chrom, pos = np.asarray(chrom), np.asarray(pos)
    out = np.zeros(len(pos), dtype=bool)
    for name in np.unique(chrom):
        if name in union:
            rows = np.flatnonzero(chrom == name)
            start, end = union[name]
            index = np.searchsorted(start, pos[rows], side="right") - 1
            out[rows] = (index >= 0) & (pos[rows] < end[np.maximum(index, 0)])
    return out


def is_start_codon(genome: dict, chrom: np.ndarray, first: np.ndarray, strand: np.ndarray) -> np.ndarray:
    """Whether the three bases beginning at ``first`` (first coding base, transcript orientation) read ATG."""
    out = np.zeros(len(first), dtype=bool)
    for name in np.unique(chrom):
        rows = np.flatnonzero(chrom == name)
        seq, plus = genome[name], strand[rows] == "+"
        p = np.clip(first[rows], 2, len(seq) - 3)
        triplet = seq[np.where(plus, p, p - 2)[:, None] + np.arange(3)]
        out[rows] = np.where(plus, (triplet == (0, 3, 2)).all(1), (triplet == (1, 0, 3)).all(1))
    return out


def coding_starts(cds: pd.DataFrame) -> pd.DataFrame:
    """First coding base of each gene: chrom, first, strand."""
    genes = cds.groupby("group").agg(chrom=("chrom", "first"), strand=("strand", "first"), start=("start", "min"),
                                     end=("end", "max"))
    genes["first"] = np.where(genes["strand"] == "+", genes["start"], genes["end"] - 1)
    return genes.reset_index()


def regulatory(coords: Path, dataset: str, names: dict[str, str], strip_chr: bool = False) -> pd.DataFrame:
    """Genomic Benchmarks coordinate files -> feature rows. ``names`` maps class file to sequence class."""
    parts = []
    for name, biotype in names.items():
        for split in ("train", "test"):
            rows = pd.read_csv(coords / dataset / split / f"{name}.csv.gz", dtype={"id": str})
            chrom = rows["region"].str.removeprefix("chr") if strip_chr else rows["region"]
            parts.append(table(chrom, rows["start"], rows["end"], "+", biotype, f"{dataset}:{split}:" + rows["id"]))
    return pd.concat(parts, ignore_index=True)


def human(src: dict, genome: dict):
    root = Path(src["human_bed"])
    cds, genes = bed(root / "cds.bed.gz"), bed(root / "gene.bed.gz")
    rna = []
    for chrom in genome:
        with gzip.open(Path(src["human_gbk"]) / f"{chrom}.features.gz", "rt") as handle:
            for key, strand, parts, quals in genbank_features(handle):
                biotype = HUMAN_RNA.get(quals.get("note", [""])[0]) if key == "misc_RNA" else None
                if biotype:
                    start, end = max(parts, key=lambda s: s[1] - s[0])
                    rna.append((chrom, start, end, strand, biotype, quals["gene"][0], ""))
    marks = regulatory(Path(src["coords"]), "human_ensembl_regulatory",
                       {"enhancer": "enhancer", "ocr": "open_chromatin", "promoter": "promoter"})
    introns = segments(bed(root / "intron.bed.gz"), "intron", MIN_INTRON)
    introns = introns[~covered(merged(marks), introns["chrom"], (introns["start"] + introns["end"]) // 2)]
    features = pd.concat([
        segments(cds, "CDS", MIN_SEGMENT), segments(bed(root / "utr5.bed.gz"), "five_prime_UTR", MIN_SEGMENT),
        segments(bed(root / "utr3.bed.gz"), "three_prime_UTR", MIN_SEGMENT), introns,
        tss(genes[genes["group"].isin(cds["group"])]), pd.DataFrame(rna, columns=COLUMNS),
        marks[marks["biotype"] != "promoter"]], ignore_index=True)
    return features[features["chrom"].isin(genome)], pd.concat([genes, marks])[["chrom", "start", "end"]], coding_starts(cds)


def fly(src: dict, genome: dict):
    gtf = pd.read_csv(src["fly_gtf"], sep="\t", header=None, usecols=[0, 2, 3, 4, 6, 8],
                      names=["chrom", "kind", "start", "end", "strand", "attributes"], dtype={"chrom": str})
    gtf = gtf[gtf["chrom"].isin(FLY_CHROMS)]
    gtf["start"] -= 1
    gtf["group"] = gtf["attributes"].str.extract(r'gene_id "([^"]+)"', expand=False)
    gtf["transcript"] = gtf["attributes"].str.extract(r'transcript_id "([^"]+)"', expand=False)
    kind, genes, cds = gtf["kind"], gtf[gtf["kind"] == "gene"], gtf[gtf["kind"] == "CDS"]
    exons, parts = gtf[kind == "exon"], []
    for source, biotype in FLY_EXONIC.items():
        own = exons[exons["transcript"].isin(gtf.loc[kind == source, "transcript"])]
        longest = own.loc[(own["end"] - own["start"]).groupby(own["transcript"]).idxmax()]
        parts.append(segments(longest, biotype, MIN_SEGMENT))
    parts += [segments(gtf[kind == source], biotype, 0) for source, biotype in FLY_SMALL.items()]
    introns = bed(Path(src["fly_bed"]) / "intron.bed.gz")
    features = pd.concat([
        segments(cds, "CDS", MIN_SEGMENT), segments(gtf[kind == "5UTR"], "five_prime_UTR", MIN_SEGMENT),
        segments(gtf[kind == "3UTR"], "three_prime_UTR", MIN_SEGMENT),
        segments(introns[introns["chrom"].isin(FLY_CHROMS)], "intron", MIN_INTRON),
        tss(genes[genes["group"].isin(cds["group"])]), *parts,
        regulatory(Path(src["coords"]), "drosophila_enhancers_stark", {"positive": "enhancer"}, strip_chr=True)],
        ignore_index=True)
    return features[features["chrom"].isin(FLY_CHROMS)], genes[["chrom", "start", "end"]], coding_starts(cds)


def lift_offsets(starts: pd.DataFrame, genome: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Piecewise-constant coordinate offsets between an annotation and a slightly different assembly.

    For every annotated coding start the offset in +-LIFT_RANGE that puts an ATG there is scored, the
    score is smoothed over neighbouring genes, and each gene takes the best smoothed offset, refined to
    the nearest exact hit. Returns chrom -> (anchor positions, offsets).
    """
    deltas, out = np.arange(-LIFT_RANGE, LIFT_RANGE + 1), {}
    for chrom, part in starts.sort_values("first").groupby("chrom"):
        first, strand = part["first"].to_numpy(), part["strand"].to_numpy()
        keep = (first > LIFT_RANGE + 3) & (first < len(genome[chrom]) - LIFT_RANGE - 3)
        first, strand = first[keep], strand[keep]
        hit = np.stack([is_start_codon(genome, np.full(len(first), chrom), first + d, strand) for d in deltas])
        best = uniform_filter1d(hit.astype(np.float32), LIFT_SMOOTH, axis=1, mode="nearest").argmax(0)
        near = best[None, :] + np.arange(-LIFT_REFINE, LIFT_REFINE + 1)[:, None]
        exact = hit[np.clip(near, 0, len(deltas) - 1), np.arange(len(first))]
        cost = np.where(exact, np.abs(near - best), LIFT_REFINE + 1)
        refined = np.where(exact.any(0), near[cost.argmin(0), np.arange(len(first))], best)
        out[chrom] = (first, deltas[np.clip(refined, 0, len(deltas) - 1)])
    return out


def lifted(rows: pd.DataFrame, offsets: dict) -> pd.DataFrame:
    """Shift each row by the offset of the nearest anchor on its chromosome."""
    rows = rows.copy()
    for chrom, (anchors, delta) in offsets.items():
        mask = (rows["chrom"] == chrom).to_numpy()
        mid = ((rows.loc[mask, "start"] + rows.loc[mask, "end"]) // 2).to_numpy()
        right = np.clip(np.searchsorted(anchors, mid), 1, len(anchors) - 1)
        nearest = np.where(mid - anchors[right - 1] <= anchors[right] - mid, right - 1, right)
        rows.loc[mask, ["start", "end"]] = rows.loc[mask, ["start", "end"]].to_numpy() + delta[nearest][:, None]
    return rows


def worm(src: dict, genome: dict):
    """WBcel235 (Ensembl 96) annotation on the ce10 (WS220) sequence, lifted by local start-codon alignment."""
    root = Path(src["worm_bed"])
    raw = [bed(root / f"{name}.bed.gz") for name in ("cds", "utr5", "utr3", "intron", "gene")]
    raw = [rows[rows["chrom"].isin(genome)] for rows in raw]
    offsets = lift_offsets(coding_starts(raw[0]), genome)
    cds, utr5, utr3, introns, genes = (lifted(rows, offsets) for rows in raw)
    features = pd.concat([
        segments(cds, "CDS", MIN_SEGMENT), segments(utr5, "five_prime_UTR", MIN_SEGMENT),
        segments(utr3, "three_prime_UTR", MIN_SEGMENT), segments(introns, "intron", MIN_INTRON),
        tss(genes[genes["group"].isin(cds["group"])])], ignore_index=True)
    return features, genes[["chrom", "start", "end"]], coding_starts(cds)


def arabidopsis(src: dict, genome: dict):
    root = Path(src["arabidopsis_dir"])
    rows = pd.read_csv(root / "all_exons.txt", sep=" ", quotechar='"', skiprows=1, header=None,
                       names=["row", "rank", "reverse_rank", "chrom", "kind", "start", "end", "strand", "name"])
    rows["start"] -= 1
    rows["end"] = rows["end"].astype(int)
    genes = rows[rows["kind"] == "gene"].copy()
    genes["group"] = genes["name"].str.extract(r"ID=([^;]+)", expand=False)
    genes["type"] = genes["name"].str.extract(r"Note=(\w+)", expand=False)
    coding = genes[genes["type"] == "protein_coding_gene"]
    exons = rows[rows["kind"] == "exon"].copy()
    exons["transcript"] = exons["name"].str.removeprefix("Parent=")
    exons["group"] = exons["transcript"].str.split(".").str[0]
    internal = exons[(exons["rank"] > 1) & (exons["reverse_rank"] > 1) & exons["transcript"].str.endswith(".1")
                     & exons["group"].isin(coding["group"])]
    introns = pd.concat([pd.read_csv(root / f"all_{strand}_introns.txt", sep=" ", quotechar='"', skiprows=1, header=None,
                                     names=["row", "chrom", "rank", "reverse_rank", "start", "end", "name"]).assign(
        strand=sign) for strand, sign in (("plus", "+"), ("minus", "-"))])
    introns["chrom"], introns["end"] = "Chr" + introns["chrom"].astype(str), introns["end"] - 1
    introns["group"] = introns["name"].str.removeprefix("Parent=").str.split(".").str[0]
    repeats = pd.read_csv(root / "TAIR10_Buisine_TEs_strand_tab_ann.txt", sep="\t", header=0,
                          names=["chrom", "start", "end", "strand", "group", "family", "detail"])
    repeats["start"] -= 1
    features = pd.concat([
        segments(internal, "CDS", MIN_SEGMENT), segments(introns, "intron", MIN_INTRON), tss(coding),
        segments(genes[genes["type"] == "tRNA"], "tRNA", 0), segments(repeats, "transposable_element", MIN_REPEAT)],
        ignore_index=True)
    return features[features["chrom"].isin(genome)], pd.concat([genes, repeats])[["chrom", "start", "end"]], None


def compact(records, promoter_offset: int, clearance: int):
    """Feature table and coding starts of a compact genome from GenBank records (yeast chromosomes, the E. coli chromosome).

    No transcription start sites are annotated, so "promoter" windows are centred ``promoter_offset`` bases
    upstream of a start codon, and kept only if no other gene lies within ``clearance`` bases upstream.
    """
    rows = []
    for chrom, text in records:
        for key, strand, parts, quals in genbank_features(io.StringIO(text)):
            if not parts:
                continue
            start, end = max(parts, key=lambda s: s[1] - s[0])
            name = (quals.get("locus_tag") or quals.get("gene") or [f"{chrom}:{start}"])[0]
            if key in ("CDS", "gene"):
                rows.append((chrom, parts[0][0], parts[-1][1], strand, key, name, ""))
            elif key == "tRNA":
                rows.append((chrom, start, end, strand, "tRNA", name, quals.get("product", [""])[0]))
            elif key in ("LTR", "mobile_element") and end - start >= MIN_REPEAT:
                detail = quals.get("mobile_element_type", quals.get("note", [""]))[0].split(";")[0]
                rows.append((chrom, start, end, strand, "transposable_element", name, detail))
            elif key == "rep_origin":
                rows.append((chrom, start, end, strand, "replication_origin", name, quals.get("note", [""])[0].split(";")[0]))
    everything = pd.DataFrame(rows, columns=COLUMNS)
    genes, cds = everything[everything["biotype"] == "gene"], everything[everything["biotype"] == "CDS"]
    starts = coding_starts(cds)
    sign = np.where(starts["strand"] == "+", -1, 1)
    union = merged(pd.concat([genes, everything[everything["biotype"] == "tRNA"]]))
    clear = ~(covered(union, starts["chrom"], starts["first"] + sign * promoter_offset)
              | covered(union, starts["chrom"], starts["first"] + sign * clearance))
    anchors = (starts["first"] + sign * promoter_offset)[clear]
    promoters = table(starts["chrom"][clear], anchors, anchors + 1, starts["strand"][clear], "promoter", starts["group"][clear])
    features = pd.concat([segments(cds, "CDS", MIN_SEGMENT), promoters,
                          everything[~everything["biotype"].isin(("gene", "CDS"))]], ignore_index=True)
    return features, genes[["chrom", "start", "end"]], starts


def yeast(src: dict, genome: dict):
    return compact(yeast_records(src["yeast_zip"]), promoter_offset=100, clearance=150)


def ecoli(src: dict, genome: dict):
    return compact([("chr", Path(src["ecoli_gb"]).read_text())], promoter_offset=50, clearance=100)


BUILDERS = {"ecoli": ecoli, "yeast": yeast, "arabidopsis": arabidopsis, "worm": worm, "fly": fly, "human": human}
