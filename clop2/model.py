"""Encoders, objective and label-free inference."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ARCH = {"dim": 64, "c1": 64, "k1": 15, "pool1": 8, "c2": 96, "k2": 7, "pool2": 2, "dilations": (1, 2, 4), "core_bp": 200}
PAD = 255


def one_hot(codes: np.ndarray) -> torch.Tensor:
    """(n, L) uint8 codes -> (n, 4, L) float; any code above 3 (padding, N) becomes an all-zero column."""
    return F.one_hot(torch.from_numpy(np.minimum(codes, 4).astype(np.int64)), 5)[:, :, :4].permute(0, 2, 1).float()


def reverse_complement(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Reverse complement of right-padded one-hot rows, keeping the padding on the right."""
    width = x.shape[2]
    index = (torch.arange(width)[None, :] + (width - lengths)[:, None]) % width
    # Channel flip is the complement (ACGT -> TGCA), position flip the reversal; the roll restores right padding.
    return x.flip(1, 2).gather(2, index[:, None, :].expand(-1, 4, -1))


class DNAEncoder(nn.Module):
    """Convolutional stem (a learned k-mer tokeniser), dilated residual blocks with batch
    normalisation, then four pooled views: maximum and mean over the whole sequence, and
    maximum and mean over the central ``core_bp``.

    The whole-sequence views summarise context. The central views anchor the embedding to the
    middle of the input, so that "what sits at the centre of this window" is answerable; a purely
    global pooling cannot tell a promoter at the centre from one in the flank.
    """

    def __init__(self, dim, c1, k1, pool1, c2, k2, pool2, dilations, core_bp, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(4, c1, k1, padding=k1 // 2)
        self.conv2 = nn.Conv1d(c1, c2, k2, padding=k2 // 2)
        self.blocks = nn.ModuleList(nn.Conv1d(c2, c2, 3, padding=d, dilation=d) for d in dilations)
        self.norms = nn.ModuleList(nn.BatchNorm1d(c) for c in (c1, c2, *[c2] * len(dilations)))
        self.fc = nn.Linear(4 * c2, dim)
        self.pool1, self.pool2, self.dropout = pool1, pool2, nn.Dropout(dropout)
        self.core_reach = core_bp / 2 / (pool1 * pool2)

    def strand(self, x, lengths):
        # Positions past each row's own length are zeroed after every pooling step and residual block,
        # which makes a right-padded row numerically identical to the same sequence embedded on its own.
        steps1 = lengths // self.pool1
        steps = (steps1 // self.pool2).clamp(min=1)
        h = F.max_pool1d(F.relu(self.norms[0](self.conv1(x))), self.pool1)
        h = h * (torch.arange(h.shape[-1])[None, :] < steps1[:, None])[:, None, :]
        h = F.max_pool1d(F.relu(self.norms[1](self.conv2(h))), self.pool2)
        position = torch.arange(h.shape[-1])[None, :]
        valid = (position < steps[:, None])[:, None, :]
        core = valid & ((position - (steps[:, None] - 1) / 2).abs() <= self.core_reach)[:, None, :]
        h = h * valid
        for block, norm in zip(self.blocks, self.norms[2:]):
            h = (h + F.relu(norm(block(h)))) * valid
        views = [h.masked_fill(~mask, -torch.inf).amax(-1) for mask in (valid, core)]
        views += [(h * mask).sum(-1) / mask.sum(-1) for mask in (valid, core)]
        return self.fc(self.dropout(torch.cat(views, 1)))

    def forward(self, x, lengths=None):
        if lengths is None:
            lengths = torch.full((len(x),), x.shape[2])
        return F.normalize(self.strand(x, lengths) + self.strand(reverse_complement(x, lengths), lengths), dim=1)


class TextEncoder(nn.Module):
    """Bag-of-words annotation encoder; unseen prompts compose from the tokens seen in training."""

    def __init__(self, vocab: dict[str, int], dim: int):
        super().__init__()
        self.vocab = vocab
        self.embedding = nn.EmbeddingBag(len(vocab), dim, mode="mean")
        self.fc = nn.Linear(dim, dim)

    def forward(self, texts) -> torch.Tensor:
        words = [[self.vocab[w] for w in t.split() if w in self.vocab] for t in texts]
        if not all(words):
            raise ValueError("A prompt has no token from the training vocabulary: " + ", ".join(sorted(self.vocab)))
        offsets = torch.tensor([0] + [len(w) for w in words[:-1]]).cumsum(0)
        return F.normalize(self.fc(self.embedding(torch.tensor([i for w in words for i in w]), offsets)), dim=1)


class Clop2(nn.Module):
    def __init__(self, vocab: dict[str, int], temperature: float = 0.07):
        super().__init__()
        self.dna, self.text = DNAEncoder(**ARCH), TextEncoder(vocab, ARCH["dim"])
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / temperature)))

    def scale(self):
        return self.logit_scale.clamp(max=math.log(100.0)).exp()


def contrastive_loss(dna, text, text_ids, scale, multi_positive=True):
    """Symmetric DNA/text InfoNCE.

    multi_positive: ``text`` holds each distinct annotation once and every DNA row sharing it is a positive.
    Otherwise (original CLOP): one text copy per row, only the diagonal is positive.
    """
    if not multi_positive:
        logits, target = scale * dna @ text[text_ids].T, torch.arange(len(dna))
        return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
    logits = scale * dna @ text.T
    positives = F.one_hot(text_ids, len(text)).T.float()
    text_to_dna = -(F.log_softmax(logits.T, dim=1) * positives).sum(1) / positives.sum(1).clamp(min=1)
    return (F.cross_entropy(logits, text_ids) + text_to_dna[positives.any(1)].mean()) / 2


@torch.no_grad()
def embed(model: Clop2, codes: np.ndarray, lengths: np.ndarray | None = None, batch: int = 256) -> np.ndarray:
    """Sequences in, unit-norm DNA embeddings out. ``lengths`` marks right-padded rows of unequal length."""
    model.eval()
    parts = [model.dna(one_hot(codes[i:i + batch]), None if lengths is None else torch.from_numpy(lengths[i:i + batch]))
             for i in range(0, len(codes), batch)]
    return torch.cat(parts).numpy()


def save(model: Clop2, path: str | Path, **extra) -> None:
    torch.save({"model": model.state_dict(), "vocab": model.text.vocab, "arch": ARCH, **extra}, path)


def load(path: str | Path) -> tuple[Clop2, dict]:
    state = torch.load(path, weights_only=True)
    model = Clop2(state["vocab"])
    model.load_state_dict(state["model"])
    return model.eval(), state
