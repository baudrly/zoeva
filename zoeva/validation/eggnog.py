import os, shutil, subprocess
from typing import Optional

def run_eggnog_mapper(fasta: str, data_dir: Optional[str], out_dir: str, cpu: int = 8) -> Optional[str]:
    os.makedirs(out_dir, exist_ok=True)
    if data_dir and shutil.which("emapper.py") is not None:
        outfile = os.path.join(out_dir, "eggnog.tsv")
        cmd = ["emapper.py", "-i", fasta, "--data_dir", data_dir, "-o", os.path.join(out_dir, "eggnog"), "--output", "tsv", "--cpu", str(cpu)]
        subprocess.run(cmd, check=True)
        return outfile
    return None
