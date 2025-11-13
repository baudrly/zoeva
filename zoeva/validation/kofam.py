import os, shutil, subprocess
from typing import Optional

def run_kofamscan(fasta: str, db_dir: Optional[str], out_dir: str, cpu: int = 8) -> Optional[str]:
    os.makedirs(out_dir, exist_ok=True)
    if db_dir and shutil.which("exec_annotation") is not None:
        outfile = os.path.join(out_dir, "kofam.tsv")
        cmd = ["exec_annotation", "-o", outfile, "-p", os.path.join(db_dir, "profiles"), "-k", os.path.join(db_dir, "ko_list"), fasta]
        subprocess.run(cmd, check=True)
        return outfile
    return None
