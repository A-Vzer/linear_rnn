r"""Multi-query associative recall (MQAR).

Capability isolated: can the layer write a set of key->value bindings into its
state and later read back the value for an arbitrary, repeatedly-queried key?
This is the classic memory-capacity probe that separates gated/delta linear RNNs
from weaker recurrences.

Knob to sweep:
  - ``n_pairs`` : number of stored bindings — push it *toward and past the layer's
                  key/state dimension* to find where recall collapses.
  - ``gap``     : number of blank filler tokens between the writes and the queries,
                  i.e. the write->query distance (tests retention over time).

Token layout (integer ids; id 0 is a reserved blank/filler, keys & values use
[1, vocab)):
  [k0 v0 ... k(n-1) v(n-1)]  [dk0 dv0 ... ]  [0 * gap]  [q0 q1 ... q(m-1)]
   \---- target writes ----/  \-distractors-/ \--gap--/  \--- query block ---/
Each query token is one of the *target* stored keys; the target at that position
is the value bound to it. All keys in a sequence (targets + distractors) are
distinct, so recall is unambiguous.

Knob ``n_distractors`` adds key->value pairs that are **never queried** and whose
keys are disjoint from the target keys. They sit *after* the target writes (so
they are the most-recent bindings before the queries): pure interference that a
single-step read-out can be overwritten by, but an exact full-history solve can
deconvolve. Because the distractors are newer than the targets, *forgetting*
(recency bias) hurts here — the opposite of the drift task — so this is also the
recall setting in which sweeping the forget gate finally has teeth.
"""

from __future__ import annotations

import numpy as np


def make_mqar(
    batch: int,
    n_pairs: int,
    n_queries: int,
    gap: int,
    vocab: int,
    seed: int | None = None,
    n_distractors: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate multi-query associative-recall sequences.

    Args:
        batch: number of independent sequences.
        n_pairs: number of *target* key->value bindings written per sequence.
        n_queries: number of recall queries appended after the gap.
        gap: count of blank (id 0) filler tokens between writes and queries.
        vocab: vocabulary size; keys/values are drawn from [1, vocab).
        seed: PRNG seed for reproducibility.
        n_distractors: number of extra key->value pairs that are **never queried**
            (keys disjoint from the target keys), written right after the target
            block so they are the most-recent bindings. Pure interference; 0 (the
            default) reproduces the plain task exactly.

    Returns:
        inputs:  int64 ``(batch, L)`` token-id stream,
                 ``L = 2*(n_pairs + n_distractors) + gap + n_queries``.
        targets: int64 ``(batch, L)`` — bound value at each query position, 0 else.
        mask:    bool  ``(batch, L)`` — True only at answer (query) positions.
    """
    if batch <= 0 or n_pairs <= 0 or n_queries <= 0:
        raise ValueError("batch, n_pairs, and n_queries must be positive.")
    if gap < 0:
        raise ValueError("gap must be non-negative.")
    if n_distractors < 0:
        raise ValueError("n_distractors must be non-negative.")
    n_keys = n_pairs + n_distractors
    if vocab - 1 < n_keys:
        raise ValueError(
            f"vocab too small: need at least {n_keys + 1} ids for {n_keys} distinct "
            f"keys (targets + distractors) plus the reserved blank, got vocab={vocab}."
        )

    rng = np.random.default_rng(seed)
    write_len = 2 * n_pairs
    dist_len = 2 * n_distractors
    seq_len = write_len + dist_len + gap + n_queries
    d_start = write_len                       # distractor block start
    q_start = write_len + dist_len + gap      # query block start

    inputs = np.zeros((batch, seq_len), dtype=np.int64)
    targets = np.zeros((batch, seq_len), dtype=np.int64)
    mask = np.zeros((batch, seq_len), dtype=bool)

    nonblank = np.arange(1, vocab)
    for b in range(batch):
        # All keys mutually distinct; first n_pairs are the (queryable) targets,
        # the rest are distractor keys (never queried).
        all_keys = rng.choice(nonblank, size=n_keys, replace=False)
        keys = all_keys[:n_pairs]
        values = rng.integers(1, vocab, size=n_pairs)

        # Target write block: interleaved key, value.
        inputs[b, 0:write_len:2] = keys
        inputs[b, 1:write_len:2] = values

        # Distractor write block (never queried; keys disjoint from targets).
        if n_distractors:
            dkeys = all_keys[n_pairs:]
            dvalues = rng.integers(1, vocab, size=n_distractors)
            inputs[b, d_start:d_start + dist_len:2] = dkeys
            inputs[b, d_start + 1:d_start + dist_len:2] = dvalues
        # gap positions stay 0 (blank filler).

        # Query block: sample which TARGET keys to ask about (with repeats).
        which = rng.integers(0, n_pairs, size=n_queries)
        inputs[b, q_start:q_start + n_queries] = keys[which]
        targets[b, q_start:q_start + n_queries] = values[which]
        mask[b, q_start:q_start + n_queries] = True

    return inputs, targets, mask


if __name__ == "__main__":
    n_pairs, n_queries, gap, n_distractors = 3, 2, 2, 2
    inputs, targets, mask = make_mqar(
        batch=2, n_pairs=n_pairs, n_queries=n_queries, gap=gap, vocab=16,
        seed=0, n_distractors=n_distractors,
    )
    print("=== multi-query associative recall (with distractors) ===")
    print(f"inputs {inputs.shape}  targets {targets.shape}  mask {mask.shape}")
    print("\nSequence for example 0 (id 0 = blank filler):")
    print("  tokens :", inputs[0].tolist())
    print("  target :", targets[0].tolist())
    print("  mask   :", mask[0].astype(int).tolist())
    d_start = 2 * n_pairs
    print("\n  target writes (queryable) :",
          {int(inputs[0, 2 * i]): int(inputs[0, 2 * i + 1]) for i in range(n_pairs)})
    print("  distractor writes (NEVER queried) :",
          {int(inputs[0, d_start + 2 * i]): int(inputs[0, d_start + 2 * i + 1])
           for i in range(n_distractors)})
    for p in np.where(mask[0])[0]:
        print(f"  query @pos {p}: key {int(inputs[0, p])} -> "
              f"expected value {int(targets[0, p])}  (a target, not a distractor)")
