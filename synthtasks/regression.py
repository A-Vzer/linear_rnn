"""In-context linear regression.

Capability isolated: can the layer infer an unseen linear map ``y = W x`` purely
from (x, y) demonstrations held in its state, and apply it to a fresh query?
This is the canonical "test-time regression" probe for MesaNet / DeltaNet.

Knob to sweep:
  - ``n_examples`` : amount of in-context evidence (error should fall as it grows).
  - ``noise``      : Gaussian label noise -> irreducible error floor.
  - ``drift``      : non-stationarity of W across the sequence (0 = stationary);
                     tests tracking of a slowly changing target.

Encoding (D = d + 1 channels per token), interleaved x/y pairs:
  even position 2i   -> x token : [ x_i (d dims) , 0 ]   <- a query position
  odd  position 2i+1 -> y token : [ 0 (d dims)   , y_i ] <- reveals the answer
At each x position the model must predict y_i *before* the following y token
reveals it, so prediction is strictly causal.
"""

from __future__ import annotations

import numpy as np


def make_regression(
    batch: int,
    n_examples: int,
    d: int,
    noise: float = 0.0,
    drift: float = 0.0,
    seed: int | None = None,
    return_weights: bool = False,
) -> tuple[np.ndarray, ...]:
    """Generate interleaved in-context linear-regression sequences.

    Args:
        batch: number of independent sequences.
        n_examples: number of (x, y) demonstrations per sequence.
        d: input dimension of x (token dim is ``D = d + 1``).
        noise: std of additive Gaussian label noise on y (0 = clean).
        drift: per-step std of the random-walk perturbation applied to W
            (0 = stationary). When > 0, ||W|| is held constant so only the
            *direction* of W drifts.
        seed: PRNG seed for reproducibility.
        return_weights: if True, also return the per-example ground-truth weight
            trajectory ``W`` (so callers can inspect how drift moves the map).

    Returns:
        inputs:  float64 ``(batch, 2*n_examples, d + 1)`` interleaved x/y tokens.
        targets: float64 ``(batch, 2*n_examples)`` — y_i at x positions, 0 else.
        mask:    bool    ``(batch, 2*n_examples)`` — True at x (query) positions.
        weights: float64 ``(batch, n_examples, d)`` — the W used at each example;
            ONLY returned when ``return_weights=True`` (4-tuple instead of 3).
    """
    if batch <= 0 or n_examples <= 0 or d <= 0:
        raise ValueError("batch, n_examples, and d must all be positive.")

    rng = np.random.default_rng(seed)

    # Inputs x_i ~ N(0, I_d).
    x = rng.standard_normal((batch, n_examples, d))

    # Weight trajectory W[:, i] per sequence. Scale by 1/sqrt(d) so Var(y) ~ 1.
    weights = np.empty((batch, n_examples, d))
    weights[:, 0] = rng.standard_normal((batch, d)) / np.sqrt(d)
    norm0 = np.linalg.norm(weights[:, 0], axis=-1, keepdims=True)
    for i in range(1, n_examples):
        step = rng.standard_normal((batch, d)) * drift
        w_i = weights[:, i - 1] + step
        if drift > 0.0:
            # Keep magnitude fixed so drift rotates rather than rescales W.
            w_i = w_i / np.linalg.norm(w_i, axis=-1, keepdims=True) * norm0
        weights[:, i] = w_i

    # Labels y_i = W_i . x_i (+ noise).
    y = np.sum(weights * x, axis=-1)  # (batch, n_examples)
    if noise > 0.0:
        y = y + rng.standard_normal(y.shape) * noise

    # Assemble interleaved sequence of length L = 2 * n_examples.
    seq_len = 2 * n_examples
    inputs = np.zeros((batch, seq_len, d + 1))
    inputs[:, 0::2, :d] = x  # x tokens at even positions
    inputs[:, 1::2, d] = y   # y tokens at odd positions (scalar in last channel)

    targets = np.zeros((batch, seq_len))
    targets[:, 0::2] = y

    mask = np.zeros((batch, seq_len), dtype=bool)
    mask[:, 0::2] = True

    if return_weights:
        return inputs, targets, mask, weights
    return inputs, targets, mask


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    # --- clean, stationary reference: the worked example ---
    inputs, targets, mask = make_regression(
        batch=2, n_examples=3, d=2, noise=0.0, drift=0.0, seed=0
    )
    print("=== in-context linear regression (clean, stationary) ===")
    print(f"inputs {inputs.shape}  targets {targets.shape}  mask {mask.shape}")
    print("\nSequence for example 0 (token-dim = d+1 = 3; last channel holds y):")
    for t in range(inputs.shape[1]):
        kind = "x" if mask[0, t] else "y"
        scored = "  <- SCORED query, target y = %+.3f" % targets[0, t] if mask[0, t] else ""
        print(f"  pos {t} [{kind}]  token = {np.round(inputs[0, t], 3)}{scored}")
    print("\nReading it: each x token should predict the y revealed one step later.")

    # --- noise IS applied: same seed -> identical x and W, only y is perturbed ---
    # (the noise draw is the last RNG call, so x/W are byte-identical to the clean run.)
    NOISE = 0.5
    _, y_clean, msk = make_regression(batch=4, n_examples=8, d=4, noise=0.0, seed=1)
    _, y_noisy, _ = make_regression(batch=4, n_examples=8, d=4, noise=NOISE, seed=1)
    added = (y_noisy - y_clean)[msk]  # the injected label noise on scored targets only
    q0 = msk[0]
    print(f"\n=== noise (std={NOISE}): same seed, so x and W match; only y shifts ===")
    print(f"  clean targets (seq0, queries): {y_clean[0, q0]}")
    print(f"  noisy targets (seq0, queries): {y_noisy[0, q0]}")
    print(f"  injected noise over all queries: mean={added.mean():+.3f}  std={added.std():.3f}"
          f"  (expected std ~ {NOISE})")

    # --- drift IS applied: W rotates across the sequence, so no single map fits ---
    # Fit one OLS map on the first k pairs, then watch its error grow along the sequence.
    print("\n=== drift: a map fit on early pairs degrades along the sequence ===")
    print("  (fit one OLS map on the first k=8 (x, y) pairs, then |error| early vs late)")
    DRIFT, k = 0.15, 8
    for label, dr in (("static (drift=0.0)", 0.0), (f"drift={DRIFT}", DRIFT)):
        inp, _, _ = make_regression(batch=1, n_examples=40, d=4, noise=0.0, drift=dr, seed=2)
        X, Y = inp[0, 0::2, :4], inp[0, 1::2, 4]  # (n, d) inputs, (n,) labels
        w_hat, *_ = np.linalg.lstsq(X[:k], Y[:k], rcond=None)
        err = np.abs(X @ w_hat - Y)
        print(f"  {label:18s}: |err| first {k} = {err[:k].mean():.3f}   last {k} = {err[-k:].mean():.3f}")
    print("  -> static stays ~0 (one map fits all); drift grows (W has rotated away).")
