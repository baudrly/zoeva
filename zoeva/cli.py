import typer, os
from rich import print
from .config import EmbedConfig, HypothesisConfig, ensure_dir
from .embed import run_clop_export
from .hypothesis import generate_hypotheses
from .validation import run_blast_or_diamond, run_hmmer_scan, run_kofamscan, run_eggnog_mapper, summarize_validation
from .scoring import score_hypotheses
from .report import render_simple_report
from .graph import GraphConfig, propagate

app = typer.Typer(add_completion=False, no_args_is_help=True)

@app.command()
def embed(clop_repo: str = typer.Option(..., help="Path to cloned baudrly/clop repo"),
          input: str = typer.Option(..., help="Input FASTA/GFF3/BED"),
          input_type: str = typer.Option("fasta", help="fasta|gff3|bed"),
          reference_fasta: str = typer.Option(None, help="Reference FASTA for GFF3/BED"),
          out: str = typer.Option("run/embeddings", help="Output directory"),
          kmer_k: int = 6, epochs: int = 5, batch_size: int = 64):
    cfg = EmbedConfig(clop_repo=clop_repo, input_file=input, input_type=input_type, reference_fasta=reference_fasta,
                      output_dir=out, kmer_k=kmer_k, epochs=epochs, batch_size=batch_size)
    path = run_clop_export(cfg)
    print(f"[green]Embeddings exported:[/green] {path}")

@app.command()
def hypothesize(embeddings: str = typer.Option(..., help="final_embeddings.csv/parquet from CLOP"),
                out: str = typer.Option("run/hypotheses", help="Output dir"),
                n_neighbors: int = 50,
                use_faiss: bool = True,
                zero_shot_prompts: str = typer.Option(None, help="TXT with one prompt per line (e.g. GO:XXXX, KEGG:mapXXXXX, keywords)"),
                text_encoder_onnx: str = typer.Option(None, help="CLOP text encoder .onnx (optional)"),
                text_vocab_json: str = typer.Option(None, help="CLOP text vocab (optional)")
                ):
    cfg = HypothesisConfig(embeddings_csv=embeddings, out_dir=out, use_faiss=use_faiss, n_neighbors=n_neighbors,
                           zero_shot_prompts=zero_shot_prompts, text_encoder_onnx=text_encoder_onnx, text_vocab_json=text_vocab_json)
    path = generate_hypotheses(cfg)
    print(f"[green]Hypotheses saved:[/green] {path}")

@app.command()
def graph(hypotheses: str = typer.Option(..., help="hypotheses.parquet"),
          out: str = typer.Option("run/graph", help="Output dir"),
          kw_to_label: str = typer.Option(None, help="TSV with columns: keyword, label, weight"),
          label_edges: str = typer.Option(None, help="TSV with columns: src, dst, rel, weight"),
          alpha: float = 0.85,
          topk: int = 50):
    cfg = GraphConfig(hypotheses_parquet=hypotheses, out_dir=out, kw_to_label_tsv=kw_to_label, label_edges_tsv=label_edges, alpha=alpha, topk=topk)
    gpath = propagate(cfg)
    print(f"[green]Graph scores:[/green] {gpath}")

@app.command()
def validate(fasta: str = typer.Option(..., help="Input FASTA of query sequences"),
             out: str = typer.Option("run/validation", help="Output dir"),
             threads: int = 8,
             blast_db: str = typer.Option(None, help="BLAST DB (makeblastdb)"),
             diamond_db: str = typer.Option(None, help="DIAMOND DB"),
             pfam_hmm: str = typer.Option(None, help="Pfam-A.hmm"),
             kofam_db: str = typer.Option(None, help="KOfam DB (profiles + ko_list)"),
             eggnog_db: str = typer.Option(None, help="eggNOG data dir")):
    ensure_dir(out)
    from .validation import blast as blast_mod
    bcfg = blast_mod.BlastConfig(fasta=fasta, out_dir=out, threads=threads, blast_db=blast_db, diamond_db=diamond_db)
    _ = run_blast_or_diamond(bcfg)
    _ = run_hmmer_scan(fasta, pfam_hmm, out, cpu=threads)
    _ = run_kofamscan(fasta, kofam_db, out, cpu=threads)
    _ = run_eggnog_mapper(fasta, eggnog_db, out, cpu=threads)
    spath = summarize_validation(out)
    print(f"[green]Validation summary:[/green] {spath}")

@app.command()
def report(hypotheses: str = typer.Option(..., help="hypotheses.parquet"),
           validation: str = typer.Option(None, help="validation summary parquet"),
           graph: str = typer.Option(None, help="graph scores parquet"),
           out: str = typer.Option("run/report", help="Output dir")):
    spath = score_hypotheses(hypotheses, validation, out, graph_parquet=graph)
    rpath = render_simple_report(spath, out)
    print(f"[green]Report written:[/green] {rpath}")

if __name__ == "__main__":
    app()
