"""Training loop. Nothing outside the rows passed in is seen by the vocabulary, the weights or early stopping."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import encode_ragged
from .model import ARCH, Clop2, contrastive_loss, embed, one_hot, save

BUCKET_BATCHES = 20


def centred(codes: np.ndarray, length: int) -> np.ndarray:
    start = (codes.shape[1] - length) // 2
    return codes[:, start:start + length]


def epoch_batches(lengths: np.ndarray, batch: int, crop_lengths, rng: np.random.Generator):
    """Yield (row indices, crop length or None).

    With ``crop_lengths`` every batch gets one centred crop length, drawn independently of any label.
    Without it rows keep their own length and are bucketed so that padding stays small.
    """
    order = rng.permutation(len(lengths))
    if crop_lengths is None:
        size = batch * BUCKET_BATCHES
        order = np.concatenate([part[np.argsort(lengths[part], kind="stable")]
                                for part in np.split(order, range(size, len(order), size))])
        parts = np.split(order, range(batch, len(order), batch))
        for i in rng.permutation(len(parts)):
            yield parts[i], None
    else:
        for part in np.split(order, range(batch, len(order), batch)):
            yield part, int(rng.choice(crop_lengths))


def train(corpus: pd.DataFrame, out: str, name: str, seed: int, crop_lengths: tuple[int, ...] | None = None,
          multi_positive: bool = True, epochs: int = 10, batch: int = 256, lr: float = 1e-3) -> Path:
    """Fit CLOP2 on ``corpus`` (columns id, text, sequence).

    crop_lengths: train on random centred crops of these lengths (all sequences must share one length).
    None: train on each sequence at its own length.
    A hash-selected tenth of the rows is held back for early stopping.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    codes, lengths = encode_ragged(corpus["sequence"])
    if crop_lengths is not None and lengths.min() != lengths.max():
        raise ValueError("crop_lengths needs sequences of one common length")
    texts = sorted(corpus["text"].unique())
    text_ids = torch.from_numpy(np.searchsorted(texts, corpus["text"].to_numpy()))
    is_dev = corpus["id"].map(lambda i: int(hashlib.sha256(i.encode()).hexdigest()[:8], 16) % 10 == 0).to_numpy()
    fit, dev = np.flatnonzero(~is_dev), np.flatnonzero(is_dev)
    model = Clop2({w: i for i, w in enumerate(sorted({w for t in texts for w in t.split()}))})
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                                   total_steps=epochs * math.ceil(len(fit) / batch))
    best, history = (-1.0, None), []
    for epoch in range(epochs):
        model.train()
        for rows, length in epoch_batches(lengths[fit], batch, crop_lengths, rng):
            rows = fit[rows]
            if length is None:
                width = lengths[rows].max()
                dna = model.dna(one_hot(codes[rows, :width]), torch.from_numpy(lengths[rows]))
            else:
                dna = model.dna(one_hot(centred(codes[rows], length)))
            loss = contrastive_loss(dna, model.text(texts), text_ids[rows], model.scale(), multi_positive)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
        with torch.no_grad():
            prototypes = model.eval().text(texts).numpy()
        views = ([embed(model, codes[dev], lengths[dev])] if crop_lengths is None
                 else [embed(model, centred(codes[dev], length)) for length in crop_lengths])
        accuracy = float(np.mean([(view @ prototypes.T).argmax(1) == text_ids[dev].numpy() for view in views]))
        history.append({"epoch": epoch + 1, "last_batch_loss": loss.item(), "dev_text_accuracy": accuracy})
        print(name, history[-1], flush=True)
        if accuracy > best[0]:
            best = (accuracy, {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save(model, out_dir / f"{name}.pt", texts=texts, provenance={
        "name": name, "objective": "multi_positive" if multi_positive else "one_positive_per_row", "arch": ARCH,
        "parameters": sum(p.numel() for p in model.parameters()), "seed": seed, "epochs": epochs, "batch": batch,
        "max_lr": lr, "crop_lengths": crop_lengths, "fit_rows": len(fit), "dev_rows": len(dev),
        "history": history, "selected_dev_text_accuracy": best[0],
        "corpus_sha256": hashlib.sha256("".join(corpus["id"]).encode()).hexdigest(),
        "exposure": "Vocabulary, weights and early stopping used only the rows passed to train().",
    })
    return out_dir / f"{name}.pt"
