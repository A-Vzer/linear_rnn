"""Task-appropriate, mask-respecting metrics.

Every metric scores ONLY the positions where ``mask`` is True, so demonstration
/ filler / non-query positions can never contribute. Predictions may be passed
either as already-decoded values or as raw class scores:
  - integer-token metrics accept logits ``(..., vocab)`` and argmax internally;
  - parity accepts floats and thresholds at 0.5.
"""

from __future__ import annotations

import numpy as np


def mse_on_queries(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Mean squared error over masked (query) positions only.

    Args:
        predictions: ``(batch, L)`` or ``(batch, L, 1)`` real-valued predictions.
        targets: ``(batch, L)`` real-valued ground truth.
        mask: ``(batch, L)`` boolean; True where the position is scored.

    Returns:
        Scalar MSE over masked positions (NaN if nothing is masked).
    """
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if predictions.shape != targets.shape:
        predictions = predictions.reshape(targets.shape)

    diff = (predictions - targets)[mask]
    if diff.size == 0:
        return float("nan")
    return float(np.mean(diff ** 2))


def mqar_exact_match(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Exact-match accuracy over answer tokens only.

    Args:
        predictions: ``(batch, L)`` token ids, or ``(batch, L, vocab)`` logits.
        targets: ``(batch, L)`` ground-truth token ids.
        mask: ``(batch, L)`` boolean; True at answer positions.

    Returns:
        Fraction of masked positions where the predicted id equals the target.
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    mask = np.asarray(mask, dtype=bool)
    if predictions.ndim == targets.ndim + 1:
        predictions = predictions.argmax(axis=-1)

    correct = (predictions == targets)[mask]
    if correct.size == 0:
        return float("nan")
    return float(np.mean(correct))


def parity_per_position_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Per-position parity accuracy over masked positions.

    Args:
        predictions: ``(batch, L)`` bits/probabilities, or ``(batch, L, 2)`` logits.
        targets: ``(batch, L)`` ground-truth parity bits in {0, 1}.
        mask: ``(batch, L)`` boolean; True where scored.

    Returns:
        Fraction of masked positions predicted correctly.
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    mask = np.asarray(mask, dtype=bool)
    if predictions.ndim == targets.ndim + 1:
        predictions = predictions.argmax(axis=-1)
    elif np.issubdtype(predictions.dtype, np.floating):
        predictions = (predictions >= 0.5).astype(np.int64)

    correct = (predictions == targets)[mask]
    if correct.size == 0:
        return float("nan")
    return float(np.mean(correct))


if __name__ == "__main__":
    # Tiny self-checks: a perfect predictor scores perfectly, and unmasked
    # positions are ignored even when wrong.
    tgt = np.array([[1.0, 2.0, 3.0, 4.0]])
    msk = np.array([[True, False, True, False]])
    perfect = tgt.copy()
    junk = np.array([[1.0, 999.0, 3.0, 999.0]])  # wrong only where unmasked
    print("=== metrics self-check ===")
    print("mse perfect prediction        :", mse_on_queries(perfect, tgt, msk))
    print("mse junk-but-only-on-unmasked :", mse_on_queries(junk, tgt, msk))

    t = np.array([[5, 0, 7]])
    m = np.array([[True, False, True]])
    p = np.array([[5, 3, 7]])
    print("mqar exact match (masked=2/2) :", mqar_exact_match(p, t, m))

    pt = np.array([[0, 1, 0, 1]])
    pm = np.ones_like(pt, dtype=bool)
    pp = np.array([[0.1, 0.9, 0.2, 0.8]])  # floats -> thresholded
    print("parity accuracy (floats)      :", parity_per_position_accuracy(pp, pt, pm))
