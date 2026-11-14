"""CLOP2: contrastive DNA / annotation-text pre-training, rebuilt from the CLOP design.

Kept from CLOP: two encoders, a shared embedding space, a CLIP objective with a learned
temperature, annotation text assembled from biotype and description fields, FASTA headers
in the ``>id|species=..|biotype=..|description=..`` convention, embedding export.

Changed: identical annotation texts are positives; the DNA encoder is convolutional,
reverse-complement invariant and accepts any length with exact padding masks; training
lengths can be randomised independently of the label; evaluation data never touches the
vocabulary, the weights or model selection; exports hold DNA embeddings and label-text
prototypes separately.
"""
from . import data  # noqa: F401
from .model import ARCH, Clop2, contrastive_loss, embed, load, one_hot, save  # noqa: F401
