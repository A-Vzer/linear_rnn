"""Running parity.

Capability isolated: maintaining a single bit of recurrent state that must be
updated at *every* step and never decay — the minimal long-range state-tracking
test. A layer that cannot integrate over the whole prefix fails this.

Knob to sweep:
  - ``seq_len`` : length over which the parity bit must be carried. Accuracy at
                  late positions reveals how far state survives.

Labels are per-position: target[t] = parity (XOR / sum mod 2) of inputs[0..t].
Every position is scored, so the mask is all-True (returned for API symmetry).
"""

from __future__ import annotations

import numpy as np


def make_parity(
    batch: int,
    seq_len: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate running-parity sequences over random binary strings.

    Args:
        batch: number of independent sequences.
        seq_len: length of each binary string.
        seed: PRNG seed for reproducibility.

    Returns:
        inputs:  int64 ``(batch, seq_len)`` random bits in {0, 1}.
        targets: int64 ``(batch, seq_len)`` running parity = cumsum(inputs) mod 2.
        mask:    bool  ``(batch, seq_len)`` — all True (every position scored).
    """
    if batch <= 0 or seq_len <= 0:
        raise ValueError("batch and seq_len must be positive.")

    rng = np.random.default_rng(seed)
    inputs = rng.integers(0, 2, size=(batch, seq_len), dtype=np.int64)
    targets = np.cumsum(inputs, axis=1) % 2
    mask = np.ones((batch, seq_len), dtype=bool)
    return inputs, targets, mask


if __name__ == "__main__":
    inputs, targets, mask = make_parity(batch=2, seq_len=10, seed=0)
    print("=== running parity ===")
    print(f"inputs {inputs.shape}  targets {targets.shape}  mask {mask.shape}")
    print("\nSequence for example 0:")
    print("  bits   :", inputs[0].tolist())
    print("  parity :", targets[0].tolist(), "(running count of 1s, mod 2)")
    print("\n  check: cumulative #1s =", np.cumsum(inputs[0]).tolist())
