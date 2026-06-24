"""compare — minimal sequence model + training harness to benchmark two fla
token-mixing layers (MesaNet vs Gated DeltaNet) on the synthtasks suite.

The ONLY thing that differs between conditions is the mixing layer; backbone,
embedding/in-projection, MLP, normalization, and per-head output gating are
shared. See :mod:`compare.model` for the model and :mod:`compare.train` for the
``train_eval`` entry point.
"""

__all__ = ["model", "train"]
