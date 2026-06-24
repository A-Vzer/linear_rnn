"""synthtasks — controlled synthetic-data generators for probing sequence layers.

Each task isolates one capability and exposes its difficulty as an explicit knob:

  - regression : in-context least-squares     (knob: n_examples, noise, drift)
  - mqar       : associative recall / memory   (knob: n_pairs, gap)
  - parity     : long-range state tracking      (knob: seq_len)

All generators are pure NumPy, deterministic given ``seed``, and return
``(inputs, targets, mask)`` where ``mask`` selects the positions to be scored.
The accompanying ``metrics`` always respect that mask, and ``sanity`` provides a
closed-form solver proving the clean regression task is learnable in principle.
"""

from .regression import make_regression
from .mqar import make_mqar
from .parity import make_parity
from .metrics import (
    mse_on_queries,
    mqar_exact_match,
    parity_per_position_accuracy,
)

__all__ = [
    "make_regression",
    "make_mqar",
    "make_parity",
    "mse_on_queries",
    "mqar_exact_match",
    "parity_per_position_accuracy",
]
