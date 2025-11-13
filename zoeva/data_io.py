from typing import Iterable, Optional
from dataclasses import dataclass
from Bio import SeqIO
import re

FASTA_META_PATTERN = re.compile(r"(?:^|\|)species=([^|]+)\|biotype=([^|]+)(?:\|description=([^|]+))?")

@dataclass
class SeqRecordLite:
    id: str
    seq: str
    species: Optional[str] = None
    biotype: Optional[str] = None
    description: Optional[str] = None

def parse_fasta_with_meta(path: str) -> Iterable[SeqRecordLite]:
    for rec in SeqIO.parse(path, "fasta"):
        header = rec.description
        species = biotype = desc = None
        m = FASTA_META_PATTERN.search(header)
        if m:
            species, biotype, desc = m.group(1), m.group(2), m.group(3)
        yield SeqRecordLite(id=rec.id, seq=str(rec.seq), species=species, biotype=biotype, description=desc)
