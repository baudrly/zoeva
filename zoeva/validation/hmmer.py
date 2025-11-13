import os, shutil, subprocess
from typing import Optional

def run_hmmer_scan(fasta: str, pfam_hmm: Optional[str], out_dir: str, cpu: int = 8) -> Optional[str]:
    os.makedirs(out_dir, exist_ok=True)
    if pfam_hmm and shutil.which("hmmscan") is not None:
        domtbl = os.path.join(out_dir, "pfam.domtblout")
        cmd = ["hmmscan", "--cpu", str(cpu), "--domtblout", domtbl, pfam_hmm, fasta]
        subprocess.run(cmd, check=True)
        return domtbl
    return None
