"""Genome readers (FASTA, 2bit, Bioconductor .rda, GenBank) and an on-disk cache of uint8 chromosomes."""
from __future__ import annotations

import gzip
import re
import struct
import zipfile
from pathlib import Path

import numpy as np

from .common import N, encode_bytes

_TWOBIT = np.array([3, 1, 0, 2], dtype=np.uint8)  # 2bit packs T, C, A, G as 0..3
_BIOSTRINGS = np.full(256, N, dtype=np.uint8)
_BIOSTRINGS[[1, 2, 4, 8]] = np.arange(4, dtype=np.uint8)  # Biostrings DNA byte codes for A, C, G, T
_SHIFTS = np.array([6, 4, 2, 0], dtype=np.uint8)


def twobit(path: str | Path, keep) -> dict[str, np.ndarray]:
    """Named sequences of a UCSC .2bit file as code arrays; N blocks become N."""
    raw = Path(path).read_bytes()
    signature, _, count, _ = struct.unpack_from("<4I", raw, 0)
    if signature != 0x1A412743:
        raise ValueError(f"{path}: not a little-endian 2bit file")
    out, cursor = {}, 16
    for _ in range(count):
        size = raw[cursor]
        name = raw[cursor + 1:cursor + 1 + size].decode()
        (offset,) = struct.unpack_from("<I", raw, cursor + 1 + size)
        cursor += size + 5
        if name not in keep:
            continue
        length, n_blocks = struct.unpack_from("<2I", raw, offset)
        blocks = np.frombuffer(raw, "<u4", 2 * n_blocks, offset + 8).reshape(2, n_blocks)
        (n_masks,) = struct.unpack_from("<I", raw, offset + 8 + 8 * n_blocks)
        data = offset + 16 + 8 * n_blocks + 8 * n_masks
        packed = np.frombuffer(raw, np.uint8, (length + 3) // 4, data)
        codes = _TWOBIT[(packed[:, None] >> _SHIFTS) & 3].ravel()[:length].copy()
        for start, width in blocks.T:
            codes[start:start + width] = N
        out[name] = codes
    return out


def rda_dna(path: str | Path) -> np.ndarray:
    """The DNAString payload of an old-style BSgenome .rda file: the one large raw vector of the R serialisation."""
    raw = gzip.decompress(Path(path).read_bytes())
    for match in re.finditer(b"\x00\x00\x00\x18", raw):  # RAWSXP header, followed by a big-endian length
        (length,) = struct.unpack_from(">i", raw, match.end())
        if 100_000 < length <= len(raw) - match.end() - 4:
            return _BIOSTRINGS[np.frombuffer(raw, np.uint8, length, match.end() + 4)]
    raise ValueError(f"{path}: no sequence payload found")


def fasta(path: str | Path, keep):
    """Yield (name, codes) for the kept records of a FASTA file, one record in memory at a time."""
    name, chunks = None, []
    with (gzip.open if str(path).endswith(".gz") else open)(path, "rb") as handle:
        for line in handle:
            if line.startswith(b">"):
                if name in keep:
                    yield name, encode_bytes(b"".join(chunks))
                name, chunks = line[1:].split()[0].decode(), []
            elif name in keep:
                chunks.append(line.strip())
    if name in keep:
        yield name, encode_bytes(b"".join(chunks))


_LOCATION = re.compile(r"<?(\d+)(?:\.\.>?(\d+))?")


def genbank_features(handle):
    """Yield (key, strand, [(start, end), ...] 0-based half-open in genomic order, qualifiers) from a GenBank
    feature table. Qualifiers map name -> list of values."""
    key, parts, quals = None, [], {}

    def flush():
        location = "".join(parts)
        segments = sorted((int(a) - 1, int(b or a)) for a, b in _LOCATION.findall(location))
        return key, "-" if "complement" in location else "+", segments, quals

    inside, last = False, None
    for line in handle:
        if line.startswith("FEATURES"):
            inside = True
            continue
        if line.startswith(("ORIGIN", "//", "CONTIG")):
            break
        if not inside or len(line) < 22:
            continue
        if line[5] != " ":
            if key:
                yield flush()
            key, parts, quals, last = line[5:21].strip(), [line[21:].strip()], {}, None
        elif key:
            text = line[21:].rstrip("\n")
            if text.startswith("/"):
                name, _, value = text[1:].partition("=")
                last = quals.setdefault(name, [])
                last.append(value.strip('"'))
            elif last is None:
                parts.append(text.strip())
            else:
                last[-1] = f"{last[-1]} {text.strip().strip(chr(34))}"
    if key:
        yield flush()


def genbank_sequence(text: str) -> np.ndarray:
    return encode_bytes(re.sub(r"[\d\s/]", "", text[text.index("\nORIGIN") + 7:]).encode())


def yeast_records(archive: str | Path):
    """(chromosome name, GenBank text) for the sixteen nuclear chromosomes shipped with pygenome."""
    with zipfile.ZipFile(archive) as bundle:
        for member in sorted(n for n in bundle.namelist() if n.endswith(".gb")):
            yield member[:-3], bundle.read(member).decode()


def build(species: str, sources: dict, cache: str | Path) -> dict[str, int]:
    """Write one .npy per chromosome under cache/species and return chromosome lengths."""
    out = Path(cache) / species
    out.mkdir(parents=True, exist_ok=True)
    if species == "human":
        records = ((f"chr{name}", codes) for name, codes in fasta(sources["human_fasta"], {str(i) for i in range(1, 23)}))
    elif species == "fly":
        records = twobit(sources["fly_2bit"], {"2L", "2R", "3L", "3R", "4", "X"}).items()
    elif species == "arabidopsis":
        records = twobit(sources["arabidopsis_2bit"], {f"Chr{i}" for i in range(1, 6)}).items()
    elif species == "worm":
        records = ((name, rda_dna(Path(sources["worm_rda"]) / f"chr{name}.rda")) for name in ("I", "II", "III", "IV", "V", "X"))
    elif species == "yeast":
        records = ((name, genbank_sequence(text)) for name, text in yeast_records(sources["yeast_zip"]))
    else:
        records = [("chr", genbank_sequence(Path(sources["ecoli_gb"]).read_text()))]
    lengths = {}
    for name, codes in records:
        np.save(out / f"{name}.npy", codes)
        lengths[name] = int(len(codes))
    return lengths


def load(species: str, cache: str | Path) -> dict[str, np.ndarray]:
    """Memory-mapped chromosomes of a species built earlier."""
    return {path.stem: np.load(path, mmap_mode="r") for path in sorted((Path(cache) / species).glob("*.npy"))}
