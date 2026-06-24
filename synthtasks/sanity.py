"""Reference-solver sanity check: is the regression data learnable in principle?

This module contains NO learned model — only a closed-form ordinary-least-squares
oracle that reads each sequence exactly as :func:`make_regression` lays it out
and predicts y at the query positions. Its purpose is to separate two questions:

    "is the data correct and well-posed?"   <-  answered here
    "can a model learn it?"                  <-  answered later

If a model later struggles, a PASS here means the data is not the cause.

What it confirms:
  (a) noiseless in-context regression is solved to ~0 held-out error once there
      are enough points to identify the linear map;
  (b) the scoring/masking path is *identical* to what models will use — every
      score goes through :func:`metrics.mse_on_queries`.

Layout assumed (must match make_regression; D = d + 1 channels per token):
  even position 2i   -> x token : [ x_i (d dims) , 0 ]   (query / scored)
  odd  position 2i+1 -> y token : [ 0 (d dims)   , y_i ]
So ``d`` is inferred as ``inputs.shape[-1] - 1`` and the i-th (x, y) pair lives
at sequence positions (2i, 2i+1).
"""

from __future__ import annotations

import numpy as np

try:  # independently runnable AND importable as a package module
    from .regression import make_regression
    from .metrics import mse_on_queries
except ImportError:  # pragma: no cover - direct-script fallback
    from regression import make_regression
    from metrics import mse_on_queries


def solve_closed_form(
    inputs: np.ndarray,
    mode: str = "leave_one_out",
) -> np.ndarray:
    """Closed-form OLS oracle over the in-context (x, y) pairs of each sequence.

    Reads the interleaved layout of :func:`make_regression`, fits an ordinary
    least-squares map per sequence, and writes a predicted y back at every query
    (even) position so the result can be scored directly by
    :func:`metrics.mse_on_queries`.

    Args:
        inputs: ``(batch, 2*n_examples, d+1)`` interleaved x/y tokens.
        mode: which pairs each query is fit on —
            ``"leave_one_out"``: all OTHER pairs (genuinely held-out; default);
            ``"causal"``:        only strictly-preceding pairs (model-like);
            ``"full"``:          all pairs incl. the query (training-error fit).

    Returns:
        predictions: float64 ``(batch, 2*n_examples)`` — predicted y at each query
        (even) position, 0 at the interleaved y (odd) positions. Same shape as the
        ``targets`` returned by :func:`make_regression`.
    """
    inputs = np.asarray(inputs, dtype=float)
    if inputs.ndim != 3:
        raise ValueError(f"expected inputs of shape (batch, L, d+1), got {inputs.shape}")
    batch, seq_len, d_plus_1 = inputs.shape
    d = d_plus_1 - 1

    x = inputs[:, 0::2, :d]       # (batch, n_examples, d)
    y = inputs[:, 1::2, d]        # (batch, n_examples) — y revealed by the y token
    n_examples = x.shape[1]

    predictions = np.zeros((batch, seq_len))
    all_idx = np.arange(n_examples)
    for b in range(batch):
        for i in range(n_examples):
            if mode == "leave_one_out":
                fit = all_idx != i
            elif mode == "causal":
                fit = all_idx < i
            elif mode == "full":
                fit = np.ones(n_examples, dtype=bool)
            else:
                raise ValueError(f"unknown mode {mode!r}")

            if fit.any():
                w, *_ = np.linalg.lstsq(x[b, fit], y[b, fit], rcond=None)
                predictions[b, 2 * i] = x[b, i] @ w
            # else: no fit data -> leave prediction at 0 (under-determined)

    return predictions


def _sweep(
    grid: tuple[int, ...],
    d: int,
    batch: int,
    noise: float,
    drift: float,
    base_seed: int,
    mode: str = "leave_one_out",
) -> np.ndarray:
    """Run the oracle across a sweep of ``n_examples``; return MSE per grid point.

    Returns ``(len(grid),)`` of :func:`mse_on_queries` values, one per n_examples.
    """
    out = np.empty(len(grid))
    for k, n in enumerate(grid):
        inputs, targets, mask = make_regression(
            batch=batch, n_examples=n, d=d, noise=noise, drift=drift, seed=base_seed + k
        )
        preds = solve_closed_form(inputs, mode=mode)
        out[k] = mse_on_queries(preds, targets, mask)
    return out


def run_sanity_check(
    d: int = 5,
    batch: int = 128,
    grid: tuple[int, ...] = (2, 4, 6, 8, 16, 32, 64),
    noise: float = 0.5,
    drift: float = 0.05,
    seed: int = 0,
    tol: float = 1e-6,
) -> dict:
    """Verify the regression data is well-posed via the closed-form oracle.

    Checks, all scored through :func:`metrics.mse_on_queries`:
      * noiseless -> held-out MSE collapses to ~0 once identifiable
        (leave-one-out needs n_examples >= d+1: d points to fit, one held out);
      * noisy     -> MSE decreases with more examples and approaches the noise
        floor sigma**2 (NOT zero);
      * masking   -> mse_on_queries equals a manual masked MSE, and the mask
        matches the interleaved (even-position) layout;
      * drift     -> reported only: full-history fit should degrade vs stationary.

    Args:
        d: input dimension of x.
        batch: sequences per grid point (more = tighter noise-floor estimate).
        grid: sweep of n_examples values.
        noise: label-noise std for the noisy case (floor = noise**2).
        drift: per-step W drift for the (reported-only) drift case.
        seed: base PRNG seed; the whole check is reproducible.
        tol: "~zero" threshold for the noiseless case.

    Returns:
        Results dict with the MSE curves, the noise floor, the identifiability
        threshold, and boolean PASS flags for each checked property.
    """
    identifiable = d + 1  # leave-one-out: need d points to fit after holding one out

    noiseless = _sweep(grid, d, batch, noise=0.0, drift=0.0, base_seed=seed)
    noisy = _sweep(grid, d, batch, noise=noise, drift=0.0, base_seed=seed + 1000)
    drifted = _sweep(grid, d, batch, noise=0.0, drift=drift, base_seed=seed + 2000)

    # (a) noiseless: every identifiable point must be ~0.
    noiseless_ok = bool(
        all(m < tol for n, m in zip(grid, noiseless) if n >= identifiable)
    )

    # (b) noisy: decreasing toward floor, and a real (non-zero) floor.
    floor = noise ** 2
    valid = [m for n, m in zip(grid, noisy) if n >= identifiable]
    noisy_decreases = bool(len(valid) >= 2 and valid[-1] < valid[0])
    approaches_floor = bool(valid and 0.5 * floor <= valid[-1] <= 2.0 * floor)
    floor_is_nonzero = bool(valid and valid[-1] > 10 * tol)
    noisy_ok = noisy_decreases and approaches_floor and floor_is_nonzero

    # masking equivalence: identical scoring path + layout matches the mask.
    inp, tgt, msk = make_regression(batch=8, n_examples=8, d=d, noise=0.3, seed=seed + 7)
    preds = solve_closed_form(inp)
    manual = float(np.mean(((preds - tgt)[msk]) ** 2))
    metric = mse_on_queries(preds, tgt, msk)
    scoring_matches = bool(np.isclose(manual, metric))
    expected_mask = np.zeros_like(msk)
    expected_mask[:, 0::2] = True
    layout_matches = bool(np.array_equal(msk, expected_mask))
    masking_ok = scoring_matches and layout_matches

    # drift: report-only comparison at the largest context.
    drift_degrades = bool(drifted[-1] > noiseless[-1])

    return {
        "d": d,
        "grid": grid,
        "identifiable_at": identifiable,
        "noiseless_mse": noiseless,
        "noisy_mse": noisy,
        "drifted_mse": drifted,
        "noise": noise,
        "noise_floor": floor,
        "drift": drift,
        "pass_noiseless": noiseless_ok,
        "pass_noisy": noisy_ok,
        "pass_masking": masking_ok,
        "drift_degrades": drift_degrades,
        "scoring_matches": scoring_matches,
        "layout_matches": layout_matches,
    }


def _fmt(x: float) -> str:
    return f"{x:.3e}"


if __name__ == "__main__":
    r = run_sanity_check()
    d = r["d"]
    grid = r["grid"]
    ident = r["identifiable_at"]

    print("=== regression data sanity check (closed-form OLS oracle) ===")
    print(f"d = {d}   batch per point   noise std = {r['noise']}   "
          f"noise floor = sigma^2 = {r['noise_floor']:.3f}")
    print(f"oracle = leave-one-out OLS; identifiable once n_examples >= {ident} "
          f"(d points to fit, 1 held out)\n")

    print(" n_examples |  noiseless MSE |   noisy MSE    | note")
    print(" -----------+----------------+----------------+----------------------")
    for n, m0, m1 in zip(grid, r["noiseless_mse"], r["noisy_mse"]):
        note = "under-determined" if n < ident else ""
        print(f"   {n:>4}     |   {_fmt(m0):>10}   |   {_fmt(m1):>10}   | {note}")
    print()

    print(f" drift case (noiseless, drift={r['drift']}): "
          f"MSE @ largest n = {_fmt(r['drifted_mse'][-1])} "
          f"vs stationary {_fmt(r['noiseless_mse'][-1])}  "
          f"-> {'degrades (expected)' if r['drift_degrades'] else 'no degradation'}")
    print()

    def line(name: str, ok: bool, detail: str) -> None:
        print(f" [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    line("noiseless -> ~0", r["pass_noiseless"],
         f"held-out MSE < tol for all n >= {ident}")
    line("noisy -> floor", r["pass_noisy"],
         f"MSE decreases and approaches sigma^2={r['noise_floor']:.3f} "
         f"(final={_fmt(r['noisy_mse'][-1])}, not zero)")
    line("masking match", r["pass_masking"],
         f"mse_on_queries == manual masked MSE ({r['scoring_matches']}) "
         f"and mask == even-position layout ({r['layout_matches']})")

    all_ok = r["pass_noiseless"] and r["pass_noisy"] and r["pass_masking"]
    print()
    print(f"==> OVERALL: {'PASS — data is well-posed and scoring matches metrics' if all_ok else 'FAIL — investigate above'}")
