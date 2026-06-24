"""Reusable experiment orchestration for the MesaNet-vs-GatedDeltaNet comparison.

Keeps the notebook thin: it should only import these, run them, and plot/narrate.
Two kinds of helpers live here:

  * verify_cg_semantics(): a Triton-free, CPU-runnable correctness check on the
    Mesa CG solver, what does the conjugate-gradient count actually compute?
    (Run BEFORE trusting any comparison; see the finding it encodes below.)
  * regression_examples_sweep() / mesa_cg_sweep(): the GPU training sweeps, built
    on compare.train.train_across_eval (train across the knob, eval per setting).

VERIFIED FINDING (fla-core 0.5.1; see verify_cg_semantics):
  The CG solver cold-starts at x=0 (naive.py / chunk_cg_solver_fwd.py). Therefore
    CG=0  -> mixer output is exactly ZERO (NOT gated linear attention);
    CG=1  -> equals the GLA readout up to a per-token positive scalar;
    CG=k  -> approaches the exact (H+lambda*I)^{-1} q solve as k grows.
  So the GLA-equivalent operating point is CG=1, and CG=0 is a degenerate
  no-mixing floor. The sweeps below treat CG=0 as that floor, not as GLA.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from dataclasses import dataclass, field, replace

import numpy as np
import torch
import torch.nn.functional as F

# NOTE: einops is imported lazily inside _gla_anchor (the only user) so the GPU
# training sweeps, which never touch the CPU collapse checks, don't depend on it.

# Make the project root importable whether run as a module or a script.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from compare import cache  # noqa: E402
from compare.model import ModelConfig, SequenceModel, count_params  # noqa: E402
from compare.train import (  # noqa: E402
    _masked_loss, _to_tensors, make_regression, set_seed, train_across_eval, train_eval,
)
from synthtasks.metrics import mqar_exact_match, mse_on_queries  # noqa: E402
from synthtasks.mqar import make_mqar  # noqa: E402
from synthtasks.sanity import solve_closed_form  # noqa: E402


def oracle_mse_curve(
    eval_examples: list[int],
    d: int = 8,
    noise: float = 0.0,
    drift: float = 0.0,
    batch: int = 256,
    seed: int = 0,
    mode: str = "causal",
) -> list[float]:
    """Closed-form OLS oracle: held-out query MSE vs n_examples (achievable floor).

    Uses the same ``solve_closed_form`` + ``mse_on_queries`` path as the sanity
    check, so this floor is directly comparable to the trained-model curves.
    CPU-only (no Triton).
    """
    out = []
    for i, n in enumerate(eval_examples):
        inputs, targets, mask = make_regression(
            batch=batch, n_examples=n, d=d, noise=noise, drift=drift, seed=seed + i
        )
        preds = solve_closed_form(inputs, mode=mode)
        out.append(float(mse_on_queries(preds, targets, mask)))
    return out


# --------------------------------------------------------------------------- #
# CG-semantics verification (pure torch, no Triton -> runs on CPU/Mac)
# --------------------------------------------------------------------------- #
def _load_mesa_naive():
    """Import fla's pure-torch Mesa reference WITHOUT triggering the Triton init.

    ``import fla`` works for path resolution, but importing fla submodules pulls
    in Triton (absent on macOS). We load ops/mesa_net/naive.py straight from its
    file so the math reference is usable on CPU.
    """
    import fla  # bare import is Triton-free; only used for its install path

    path = os.path.join(os.path.dirname(fla.__file__), "ops", "mesa_net", "naive.py")
    spec = importlib.util.spec_from_file_location("mesa_naive", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Shared collapse-control anchor (used by BOTH collapse checks)
# --------------------------------------------------------------------------- #
def _collapse_inputs(batch, seq_len, num_heads, head_dim, chunk_size, lamb_value, seed):
    """Synthetic Mesa inputs for the collapse checks; fixed RNG draw order.

    Shapes: q, k, v ``(B, L, h, d)``; g ``(B, L, h)`` (log-decay <0);
    beta ``(B, L, h)`` in (0, 1); lamb ``(h, d)``. Returns
    ``(q, k, v, g, beta, lamb, chunk_size)``. Matched ``seed`` reproduces the
    *same* anchor inputs across both checks, so they share one GLA reference.
    """
    g_seed = torch.Generator().manual_seed(seed)
    B, L, h, d = batch, seq_len, num_heads, head_dim
    rand = lambda *s: torch.randn(*s, generator=g_seed)
    q, k, v = rand(B, L, h, d), rand(B, L, h, d), rand(B, L, h, d)
    g = F.logsigmoid(rand(B, L, h))                    # decay gate <0
    beta = torch.rand(B, L, h, generator=g_seed)       # in (0, 1)
    lamb = torch.full((h, d), lamb_value)              # regularizer >0
    return q, k, v, g, beta, lamb, chunk_size


def _gla_anchor(q, k, v, g, beta, chunk_size):
    """The GLA reference read-out: the chunked Mesa read-out with x := q (no solve).

    This IS the anchor the CG-step collapse uses (CG=1 matches it up to a per-token
    scalar); the Λ-collapse compares against the identical function. Returns o_gla
    ``(B, L, h, d)``.
    """
    from einops import rearrange  # lazy: only the CPU collapse checks need it

    B, L, h, d = q.shape
    C = chunk_size

    def chunk(x):
        return rearrange(x, "b (n c) h ... -> b h n c ...", c=C).float()

    qc, kc, vc, gc, bc = map(chunk, [q, k, v, g, beta])
    gc = gc.cumsum(dim=-1)
    pdec = (gc[..., None] - gc[..., None, :]).exp().tril() * bc[..., None, :]
    hkv = torch.zeros(B, h, L // C, d, d)
    cur = torch.zeros(B, h, d, d)
    cdk, cdq = (gc[..., -1, None] - gc).exp(), gc.exp()
    kp = kc * cdk[..., None] * bc[..., None]
    for i in range(L // C):
        hkv[:, :, i] = cur
        cur = cur * gc[:, :, i, -1, None, None].exp() + kp[:, :, i].transpose(-2, -1) @ vc[:, :, i]
    return rearrange(
        (qc * cdq[..., None]) @ hkv + ((qc @ kc.transpose(-1, -2)) * pdec) @ vc,
        "b h n c d -> b (n c) h d",
    )


@dataclass
class CGSemantics:
    """Result of :func:`verify_cg_semantics` (all norms relative where noted)."""

    cg0_output_norm: float          # ||o(CG=0)|| ; expect ~0
    cg1_vs_gla_parallelism: float   # std/mean of (o_cg1 / o_gla) over feature dim; ~0 => parallel
    cg30_rel_error: float           # ||o(CG=30) - o_exact|| / ||o_exact|| ; expect small
    exact_norm: float
    pass_cg0_is_zero: bool
    pass_cg1_is_gla: bool
    pass_converges_to_exact: bool

    @property
    def all_pass(self) -> bool:
        return self.pass_cg0_is_zero and self.pass_cg1_is_gla and self.pass_converges_to_exact


def verify_cg_semantics(
    batch: int = 2,
    seq_len: int = 8,
    num_heads: int = 2,
    head_dim: int = 4,
    chunk_size: int = 4,
    lamb_value: float = 0.25,
    seed: int = 0,
) -> CGSemantics:
    """Check what Mesa's CG-step count computes, against the installed kernel.

    Compares the pure-torch Mesa reference at CG in {0, 1, 30} to (i) zero,
    (ii) the gated-linear-attention readout (the solve with x:=q), and (iii) the
    exact linear-solve. Returns a :class:`CGSemantics` with PASS flags. Runs on
    CPU; needs no Triton/GPU.
    """
    m = _load_mesa_naive()
    q, k, v, g, beta, lamb, C = _collapse_inputs(
        batch, seq_len, num_heads, head_dim, chunk_size, lamb_value, seed
    )

    o_exact, _, _ = m.naive_mesa_net_exact(q, k, v, g, lamb, beta)
    o0, _, _ = m.naive_mesa_net_CG(q, k, v, g, lamb, beta, C, max_CG_iteration=0)
    o1, _, _ = m.naive_mesa_net_CG(q, k, v, g, lamb, beta, C, max_CG_iteration=1)
    o30, _, _ = m.naive_mesa_net_CG(q, k, v, g, lamb, beta, C, max_CG_iteration=30)

    # GLA reference anchor: the chunked read-out with x := q (no solve).
    o_gla = _gla_anchor(q, k, v, g, beta, C)

    ratio = o1 / (o_gla + 1e-9)
    parallelism = float((ratio.std(-1) / (ratio.abs().mean(-1) + 1e-9)).mean())
    rel_err = float((o30 - o_exact).norm() / (o_exact.norm() + 1e-9))

    return CGSemantics(
        cg0_output_norm=float(o0.norm()),
        cg1_vs_gla_parallelism=parallelism,
        cg30_rel_error=rel_err,
        exact_norm=float(o_exact.norm()),
        pass_cg0_is_zero=float(o0.norm()) < 1e-4,
        pass_cg1_is_gla=parallelism < 1e-3,
        pass_converges_to_exact=rel_err < 1e-2,
    )


@dataclass
class LambdaCollapse:
    """Result of :func:`check_lambda_collapse` (second, independent collapse path)."""

    lambda_value: float        # the (uniform) Λ magnitude used, Λ = lambda_value·I
    alpha: float               # measured constant rescale o_gla -> o_Λ ; expect ~1/Λ
    cosine: float              # cos(o_Λ, o_gla) over query positions ; expect ~1
    max_abs_diff: float        # max |α·o_gla - o_Λ| on queries (after rescaling)
    mean_abs_diff: float       # mean |α·o_gla - o_Λ| on queries (after rescaling)
    query_mse: float           # mse_on_queries(α·o_gla, o_Λ), harness scoring path
    rel_mse: float             # query_mse / signal energy on queries  (= rescaled rel-err^2)
    pass_directional: bool     # cosine >= cos_tol  (the "up to a constant rescale" claim)
    pass_rescaled: bool        # rel_mse <= rel_mse_tol

    @property
    def all_pass(self) -> bool:
        return self.pass_directional and self.pass_rescaled


def check_lambda_collapse(
    batch: int = 2,
    seq_len: int = 8,
    num_heads: int = 2,
    head_dim: int = 4,
    chunk_size: int = 4,
    lambda_lower_bound: float = 49.0,
    lambda_softplus_target: float = 1.0,
    cg_steps: int = 30,
    cos_tol: float = 0.999,
    rel_mse_tol: float = 1e-2,
    seed: int = 0,
) -> LambdaCollapse:
    r"""Second collapse path: under a large fixed Λ, Mesa degenerates to the GLA anchor.

    Mesa solves ``x_t = (H_t + Λ)^{-1} q_t`` then reads out a bilinear form linear in
    ``x_t``. For large uniform ``Λ = λ·I`` we have ``(H_t + Λ)^{-1} ≈ Λ^{-1}`` (the
    accumulated key-correlation state ``H_t`` drops out), so ``x_t ≈ q_t/λ`` and, by
    linearity of the read-out, the output collapses to ``o_gla/λ``: the *same* GLA
    reference the CG-step check uses, up to the constant rescale ``α ≈ 1/λ``.

    Reuses the shared anchor (matched ``seed`` -> identical q,k,v,g,beta) and the GLA
    read-out :func:`_gla_anchor`; scores the query-masked discrepancy through
    ``synthtasks.metrics.mse_on_queries`` (same path as the rest of the harness).
    CPU-only, no Triton/GPU.

    Args:
        lambda_lower_bound, lambda_softplus_target: build Λ via fla MesaNet's verified
            parametrization ``Λ = softplus(lambda_params) + lambda_lower_bound`` (see
            note below); defaults give Λ ≈ 50·I, the paper's magnitude.
        cg_steps: CG iterations for the large-Λ solve (well-converged; the solve is
            trivial once Λ dominates).
        cos_tol, rel_mse_tol: PASS thresholds (reuse the CG check's tolerance style).

    Returns:
        :class:`LambdaCollapse` with the discrepancy metrics and PASS flags.
    """
    # fla MesaNet parametrizes Λ per-(head, dim) as softplus(lambda_params) +
    # lambda_lower_bound, with lambda_params._no_weight_decay = True (verified against
    # fla/layers/mesa_net.py: __init__ L101-106, forward L173-174). To FREEZE Λ ≈ 50·I
    # on the *real* layer: set lambda_lower_bound≈49, init lambda_params so softplus≈1,
    # and call layer.lambda_params.requires_grad_(False). This CPU check uses the naive
    # reference (same as the CG-collapse check, the Triton layer can't run on CPU), so
    # we build the resulting Λ tensor directly via that exact formula.
    inv_softplus = math.log(math.exp(lambda_softplus_target) - 1.0)
    lambda_value = float(F.softplus(torch.tensor(inv_softplus)) + lambda_lower_bound)

    # Same anchor inputs as the CG-collapse check (matched seed); lamb_value here is a
    # placeholder, we override Λ with the large value below.
    q, k, v, g, beta, _, C = _collapse_inputs(
        batch, seq_len, num_heads, head_dim, chunk_size, lamb_value=0.25, seed=seed
    )
    o_gla = _gla_anchor(q, k, v, g, beta, C)                 # reuse the SAME anchor

    m = _load_mesa_naive()
    lamb_large = torch.full((num_heads, head_dim), lambda_value)
    o_lambda, _, _ = m.naive_mesa_net_CG(q, k, v, g, lamb_large, beta, C, max_CG_iteration=cg_steps)

    # A regression batch (matched seed) supplies the query mask + harness scoring path.
    # head/dim are flattened into the feature axis so mse_on_queries scores per (B, L).
    _, _, mask = make_regression(batch=batch, n_examples=seq_len // 2, d=head_dim,
                                 noise=0.0, seed=seed)
    P = o_gla.reshape(batch, seq_len, -1).numpy()            # GLA anchor
    T = o_lambda.reshape(batch, seq_len, -1).numpy()         # Mesa large-Λ
    qsel = mask.astype(bool)
    pg, tg = P[qsel], T[qsel]                                # (n_query, h*d)

    # constant rescaling: least-squares scalar mapping o_gla -> o_Λ (expect ~1/Λ)
    alpha = float((pg * tg).sum() / (pg * pg).sum())
    cosine = float((pg * tg).sum() / (np.linalg.norm(pg) * np.linalg.norm(tg) + 1e-12))
    diff = alpha * pg - tg
    query_mse = float(mse_on_queries(alpha * P, T, mask))
    signal = float(mse_on_queries(np.zeros_like(T), T, mask))
    rel_mse = query_mse / (signal + 1e-12)

    return LambdaCollapse(
        lambda_value=lambda_value,
        alpha=alpha,
        cosine=cosine,
        max_abs_diff=float(np.abs(diff).max()),
        mean_abs_diff=float(np.abs(diff).mean()),
        query_mse=query_mse,
        rel_mse=rel_mse,
        pass_directional=cosine >= cos_tol,
        pass_rescaled=rel_mse <= rel_mse_tol,
    )


# --------------------------------------------------------------------------- #
# Training sweeps (require CUDA + Triton; run in the notebook on a GPU box)
# --------------------------------------------------------------------------- #
@dataclass
class SweepConfig:
    """Small, easily-edited training config shared by the sweeps."""

    d: int = 8                     # regression input dim (key dimension)
    noise: float = 0.0             # clean task for the sanity check
    batch_size: int = 64
    steps: int = 400
    lr: float = 3e-3
    hidden_size: int = 128
    num_heads: int = 4
    num_layers: int = 2
    eval_batches: int = 8
    dtype: torch.dtype = torch.float32   # NOTE: use torch.bfloat16 on CUDA
    device: str | None = None
    # Equalise Mesa's decay-gate init with GDN's retentive (Mamba-style) one.
    # fla's stock Mesa init forgets ~0.5/state per token, a poor prior for
    # in-context regression that makes the comparison unfair at small budgets;
    # this seeds a_proj.bias so Mesa's initial decay is ~1 (retain history). It is
    # a disclosed, fairness-motivated choice, set to None for stock fla behaviour.
    mesa_retention_init: float | None = 4.0
    # Match GDN's *initial per-step decay* to Mesa's: σ(mesa_retention_init=4.0) ≈ 0.982
    # (the same near-1 retention). fla's stock GDN decay is heterogeneous (~0.2-1.0 per
    # head), so without this the two layers forget at different rates by default. None =
    # stock fla GDN init. (See build_mixing_layer; default None keeps notebook 01 intact.)
    gdn_retention_init: float | None = None
    # Pin (and freeze) MesaNet's ridge regularizer Λ to this value so it can be swept
    # as the regularization strength; None = trainable Λ (stock). Mesa-only; ignored by
    # GDN. (See compare.model.build_mixing_layer and notebooks 01b / 02.)
    mesa_lambda: float | None = None
    extra: dict = field(default_factory=dict)

    def train_kwargs(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "steps": self.steps,
            "lr": self.lr,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "eval_batches": self.eval_batches,
            "dtype": self.dtype,
            "device": self.device,
            "mesa_retention_init": self.mesa_retention_init,
            "gdn_retention_init": self.gdn_retention_init,
            "mesa_lambda": self.mesa_lambda,
            "task_kwargs": {"d": self.d, "noise": self.noise, **self.extra},
        }


def regression_examples_sweep(
    eval_examples: list[int],
    mesa_cg: int = 30,
    train_examples: list[int] | None = None,
    seed: int = 0,
    cfg: SweepConfig | None = None,
) -> dict[str, dict]:
    """MesaNet vs Gated DeltaNet: held-out MSE vs number of in-context examples.

    One model per condition, trained across ``train_examples`` and evaluated at
    each of ``eval_examples``. Returns ``{label: train_across_eval result}``.
    """
    cfg = cfg or SweepConfig()
    train_examples = train_examples or eval_examples
    tk = cfg.train_kwargs()
    return {
        f"MesaNet (CG={mesa_cg})": train_across_eval(
            "mesa", make_regression, train_examples, eval_examples,
            cg_steps=mesa_cg, seed=seed, **tk),
        "Gated DeltaNet": train_across_eval(
            "gated_deltanet", make_regression, train_examples, eval_examples,
            seed=seed, **tk),
    }


def mesa_cg_sweep(
    cg_grid: list[int],
    eval_examples: list[int],
    train_examples: list[int] | None = None,
    with_gdn_reference: bool = True,
    seed: int = 0,
    cfg: SweepConfig | None = None,
) -> dict:
    """Sweep Mesa CG steps; each k is one MSE-vs-examples curve.

    Returns ``{"cg_grid", "per_cg": {k: result}, "gdn": result|None}``. Per the
    verified finding, k=0 is the no-mixing floor and k=1 is the GLA-like point.
    """
    cfg = cfg or SweepConfig()
    train_examples = train_examples or eval_examples
    tk = cfg.train_kwargs()
    per_cg = {
        k: train_across_eval("mesa", make_regression, train_examples, eval_examples,
                             cg_steps=k, seed=seed, **tk)
        for k in cg_grid
    }
    gdn = None
    if with_gdn_reference:
        gdn = train_across_eval("gated_deltanet", make_regression, train_examples,
                                eval_examples, seed=seed, **tk)
    return {"cg_grid": list(cg_grid), "per_cg": per_cg, "gdn": gdn}


def mesa_lambda_sweep(
    lambda_grid: list[float],
    eval_examples: list[int],
    train_examples: list[int] | None = None,
    mesa_cg: int = 30,
    with_gdn_reference: bool = True,
    seed: int = 0,
    cfg: SweepConfig | None = None,
) -> dict:
    """Sweep Mesa's ridge regularizer Λ; each Λ is one MSE-vs-examples curve.

    A *performance* counterpart to the Step-1b correctness check: rather than
    verifying that large Λ collapses Mesa onto the GLA anchor, this trains a Mesa
    model (CG fixed at ``mesa_cg``) at each frozen Λ and measures held-out query
    MSE. Larger Λ over-regularizes the in-context least-squares solve, so MSE
    should *degrade* (rise) with Λ; the GDN reference shows where Mesa's extra
    compute stops being worth it.

    Returns ``{"lambda_grid", "per_lambda": {Λ: result}, "gdn": result|None}``.
    """
    cfg = cfg or SweepConfig()
    train_examples = train_examples or eval_examples
    tk = cfg.train_kwargs()
    tk.pop("mesa_lambda", None)   # Λ is the swept variable here; don't double-pass from cfg
    per_lambda = {
        L: train_across_eval("mesa", make_regression, train_examples, eval_examples,
                             cg_steps=mesa_cg, mesa_lambda=L, seed=seed, **tk)
        for L in lambda_grid
    }
    gdn = None
    if with_gdn_reference:
        gdn = train_across_eval("gated_deltanet", make_regression, train_examples,
                                eval_examples, seed=seed, **tk)
    return {"lambda_grid": list(lambda_grid), "per_lambda": per_lambda, "gdn": gdn}


def train_regression_model(
    *,
    noise: float,
    train_knobs: list[int],
    d: int = 8,
    drift: float = 0.0,
    layer: str = "mesa",
    cg_steps: int | None = 30,
    steps: int = 600,
    seed: int = 0,
    batch_size: int = 64,
    lr: float = 3e-3,
    hidden_size: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    mesa_retention_init: float | None = 4.0,
    gdn_retention_init: float | None = None,
    device: str | None = None,
    dtype: torch.dtype | None = None,
    use_cache: bool = True,
):
    """Train (or reload) ONE regression model across a knob distribution; return the
    live :class:`~compare.model.SequenceModel`.

    The metric-returning sweeps (:func:`compare.train.train_across_eval`) discard
    the model; notebooks that need the model itself — e.g. notebook 04, which probes
    Mesa's internal solve tensors — use this instead. Training mirrors
    ``train_across_eval`` (per-batch uniform draw of ``n_examples`` from
    ``train_knobs``, seeds ``1 + step``), and the trained weights are disk-cached by
    config (see :mod:`compare.cache`), so re-running a notebook reloads instead of
    retraining.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dt = dtype or (torch.bfloat16 if dev.type == "cuda" else torch.float32)
    set_seed(seed)
    cfg = ModelConfig(
        layer=layer, input_kind="continuous", output_kind="regression",
        input_dim=d + 1, hidden_size=hidden_size, num_heads=num_heads,
        num_layers=num_layers, cg_steps=(cg_steps if layer == "mesa" else None),
        mesa_retention_init=mesa_retention_init, gdn_retention_init=gdn_retention_init,
    )
    model = SequenceModel(cfg).to(dev).to(dt)

    cache_spec = {
        "fn": "train_regression_model",
        "layer": layer,
        "cg_steps": (cg_steps if layer == "mesa" else None),
        "noise": noise,
        "drift": drift,
        "d": d,
        "train_knobs": [int(k) for k in train_knobs],   # order matters (RNG draw)
        "steps": steps,
        "seed": seed,
        "batch_size": batch_size,
        "lr": lr,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "dtype": str(dt),
        "mesa_retention_init": mesa_retention_init,
        "gdn_retention_init": gdn_retention_init,
    }

    if not (use_cache and cache.try_load(model, cache_spec)):
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        rng = np.random.default_rng(seed)
        model.train()
        for step in range(steps):
            n = int(train_knobs[rng.integers(len(train_knobs))])
            inp, tgt, msk = make_regression(batch=batch_size, n_examples=n, d=d,
                                            noise=noise, drift=drift, seed=1 + step)
            loss = _masked_loss(model(_to_tensors(inp, "continuous", dev, dt)),
                                tgt, msk, "mse", dev)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if use_cache:
            cache.save(model, cache_spec)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Noisy + drifting regression sweep helpers (used by notebook 02)
# --------------------------------------------------------------------------- #
def _model_label(layer: str, cg_steps: int | None) -> str:
    """Human-readable curve label for a (layer, cg_steps) condition."""
    if layer == "mesa":
        return f"Mesa (CG={cg_steps})"
    if layer == "gated_deltanet":
        return "Gated DeltaNet"
    return layer


def noise_drift_sweep(
    model_specs: list[tuple[str, int | None]],
    noise_grid: list[float],
    drift_grid: list[float],
    n_examples: int = 32,
    seed: int = 0,
    cfg: SweepConfig | None = None,
) -> list[dict]:
    """Train one model per (layer, cg_steps, noise, drift) cell; eval held-out per cell.

    Uses :func:`compare.train.train_eval` at a *fixed* ``n_examples``, the swept
    difficulty axes here are ``noise`` and ``drift``, not context length, and scores
    held-out batches through ``synthtasks.metrics.mse_on_queries`` (the shared path).
    ``cg_steps`` is applied only to Mesa (ignored for other layers). ``cfg`` supplies
    model size / steps / dtype / device / mesa_retention_init via :class:`SweepConfig`.

    Args:
        model_specs: list of ``(layer, cg_steps)`` conditions, e.g.
            ``[("mesa", 1), ("mesa", 30), ("gated_deltanet", None)]``.
        noise_grid, drift_grid: label-noise stds and per-step drift rates to cross.
        n_examples: fixed in-context example count for every cell.
        seed: reproducibility seed (model init + offset data batches).
        cfg: training/model config; defaults to :class:`SweepConfig`.

    Returns:
        Tidy rows (one per cell):
        ``{label, layer, cg_steps, noise, drift, n_examples, mse, num_params, device}``.
    """
    cfg = cfg or SweepConfig()
    kw = cfg.train_kwargs()
    kw.pop("task_kwargs")  # rebuilt per cell so noise/drift vary, d stays fixed
    rows: list[dict] = []
    for layer, cg in model_specs:
        for noise in noise_grid:
            for drift in drift_grid:
                res = train_eval(
                    layer, make_regression, n_examples,
                    cg_steps=(cg if layer == "mesa" else None),
                    seed=seed,
                    task_kwargs={"d": cfg.d, "noise": float(noise), "drift": float(drift)},
                    return_details=True,
                    **kw,
                )
                rows.append({
                    "label": _model_label(layer, cg),
                    "layer": layer, "cg_steps": cg,
                    "noise": float(noise), "drift": float(drift),
                    "n_examples": n_examples,
                    "mse": float(res.metric),
                    "num_params": res.num_params, "device": res.device,
                })
    return rows


def oracle_mse_sweep(
    param: str,
    grid: list[float],
    d: int = 8,
    n_examples: int = 32,
    noise: float = 0.0,
    drift: float = 0.0,
    mode: str = "leave_one_out",
    batch: int = 256,
    seed: int = 0,
) -> list[float]:
    """Closed-form OLS oracle MSE as ``param`` ('noise'|'drift') sweeps ``grid``.

    Other knobs are held fixed. Reuses ``sanity.solve_closed_form`` +
    ``metrics.mse_on_queries`` (the exact path the data sanity check uses), giving the
    irreducible-error floor the trained-model curves are read against. ``mode`` defaults
    to ``'leave_one_out'``, sanity.py's oracle, whose MSE sits at ~``noise**2`` (the
    true floor); ``'causal'`` would instead blow up at the early under-determined query
    positions. CPU-only (no Triton).
    """
    if param not in ("noise", "drift"):
        raise ValueError("param must be 'noise' or 'drift'")
    out: list[float] = []
    for i, v in enumerate(grid):
        kw = {"noise": noise, "drift": drift}
        kw[param] = float(v)
        inputs, targets, mask = make_regression(
            batch=batch, n_examples=n_examples, d=d, seed=seed + i, **kw)
        preds = solve_closed_form(inputs, mode=mode)
        out.append(float(mse_on_queries(preds, targets, mask)))
    return out


def _draw_mse_curves(ax, rows, x, *, lw, ms, label=True, dim_alpha=0.25, ykey="mse"):
    """Draw one metric-vs-``x`` line per model label onto ``ax``; shared plotting core.

    ``ykey`` selects the row field plotted on y (``"mse"`` for regression, ``"acc"`` for
    MQAR). Rows whose ``x`` value is None (e.g. GDN's ``cg_steps``) become a horizontal
    reference line. ``label=False`` suppresses legend labels (for the 2nd panel of a
    broken-axis plot, so each curve is legended once). To cut clutter, the *intermediate*
    Mesa-CG curves are drawn at ``dim_alpha`` so only the lowest- and highest-CG curves
    (the envelope) stay fully opaque; ``dim_alpha=1.0`` disables this. Returns the ordered
    label list.
    """
    mesa_cgs = sorted({r["cg_steps"] for r in rows
                       if r.get("layer") == "mesa" and r.get("cg_steps") is not None})
    outer_cgs = {mesa_cgs[0], mesa_cgs[-1]} if mesa_cgs else set()
    labels: list[str] = []
    for r in rows:
        if r["label"] not in labels:
            labels.append(r["label"])
    for lb in labels:
        pts = [r for r in rows if r["label"] == lb]
        leg = lb if label else "_nolegend_"
        is_gdn = (pts[0].get("layer") == "gated_deltanet")   # GDN = dashed red, everywhere
        if all(r.get(x) is None for r in pts):  # constant reference (e.g. GDN on a CG axis)
            ax.axhline(float(np.mean([r[ykey] for r in pts])), ls="--", lw=lw,
                       color=("red" if is_gdn else None),
                       label=(f"{lb} (reference)" if label else "_nolegend_"))
            continue
        ref = pts[0]
        is_inner_mesa = (ref.get("layer") == "mesa"
                         and ref.get("cg_steps") is not None
                         and ref["cg_steps"] not in outer_cgs)
        alpha = dim_alpha if is_inner_mesa else 1.0
        pts = sorted((r for r in pts if r.get(x) is not None), key=lambda r: r[x])
        ax.plot([r[x] for r in pts], [r[ykey] for r in pts], marker="o",
                linestyle=("--" if is_gdn else "-"), color=("red" if is_gdn else None),
                label=leg, lw=lw, ms=ms, alpha=alpha)
    return labels


def plot_mse_vs(
    rows: list[dict],
    x: str,
    *,
    ax=None,
    oracle: tuple[list[float], list[float]] | None = None,
    oracle_label: str = "closed-form oracle floor",
    title: str | None = None,
    xlabel: str | None = None,
    logy: bool = True,
    logx: bool = False,
    lw: float = 1.0,
    ms: float = 4.0,
    dim_alpha: float = 0.25,
    ykey: str = "mse",
    ylabel: str | None = None,
):
    """Plot held-out metric vs ``x`` ('noise'|'drift'|'cg_steps'|...), one line per label.

    ``ykey`` picks the plotted row field, ``"mse"`` (regression, default) or ``"acc"``
    (MQAR); ``ylabel`` overrides the y-axis label (defaults to the MSE label). Rows whose
    ``x`` value is None (e.g. GDN has ``cg_steps=None`` on a 'cg_steps' axis) are drawn as
    a horizontal reference line. ``oracle=(xs, ys)`` overlays a dashed reference (e.g.
    :func:`oracle_mse_sweep`). Intermediate Mesa-CG curves are drawn at ``dim_alpha`` so
    only the lowest/highest-CG envelope stays opaque (set 1.0 to disable). Returns the Axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5.8, 4.2))
    _draw_mse_curves(ax, rows, x, lw=lw, ms=ms, dim_alpha=dim_alpha, ykey=ykey)
    if oracle is not None:
        ax.plot(oracle[0], oracle[1], "k--", alpha=0.7, lw=lw, label=oracle_label)
    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel or x)
    base_ylabel = ylabel if ylabel is not None else "MSE"
    ax.set_ylabel(base_ylabel + ("  (log)" if logy else ""))
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_mse_vs_broken(
    rows: list[dict],
    x: str,
    *,
    lower_ylim: tuple[float, float],
    upper_ylim: tuple[float, float],
    height_ratios: tuple[float, float] = (1.0, 3.0),
    figsize: tuple[float, float] = (5.8, 4.6),
    title: str | None = None,
    xlabel: str | None = None,
    lw: float = 0.9,
    ms: float = 4.0,
    dim_alpha: float = 0.25,
):
    """Broken (cut) y-axis MSE-vs-``x`` plot: one far-off cluster up top, detail below.

    Draws the *same* curves on two stacked axes, the top showing ``upper_ylim`` (the
    distant high-MSE cluster) and the bottom ``lower_ylim`` (the squeezed low-MSE detail)
   , with the empty span between them cut out and marked by diagonal break ticks. This
    keeps every curve on a *linear* y-scale while removing the vertical dead space that a
    single axis would waste, so the inter-curve gaps at low noise stay legible. Returns
    the ``(top_ax, bottom_ax)`` pair.
    """
    import matplotlib.pyplot as plt

    fig, (top, bot) = plt.subplots(
        2, 1, sharex=True, figsize=figsize,
        gridspec_kw={"height_ratios": list(height_ratios), "hspace": 0.08},
    )
    _draw_mse_curves(top, rows, x, lw=lw, ms=ms, label=False, dim_alpha=dim_alpha)  # legend only on bottom
    _draw_mse_curves(bot, rows, x, lw=lw, ms=ms, label=True, dim_alpha=dim_alpha)

    top.set_ylim(*upper_ylim)
    bot.set_ylim(*lower_ylim)
    # hide the facing spines and stitch the cut
    top.spines["bottom"].set_visible(False)
    bot.spines["top"].set_visible(False)
    top.tick_params(bottom=False, labelbottom=False)
    kw = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=8, linestyle="none",
              color="k", mec="k", mew=1, clip_on=False)
    top.plot([0, 1], [0, 0], transform=top.transAxes, **kw)
    bot.plot([0, 1], [1, 1], transform=bot.transAxes, **kw)

    for axis in (top, bot):
        axis.grid(True, which="both", alpha=0.3)
    bot.set_xlabel(xlabel or x)
    bot.set_ylabel("MSE")
    bot.yaxis.set_label_coords(-0.1, 0.5 + height_ratios[0] / sum(height_ratios) / 2)
    if title:
        top.set_title(title)
    bot.legend(fontsize=8)
    return top, bot


def _draw_cg_family(ax, rows, family, *, cmap, show_gdn, lw, ms, label=True):
    """Draw the per-``family``-value CG-step curves onto ``ax``; shared plotting core.

    One solid Mesa curve per value of ``family`` (coloured by ``cmap``), with the GDN
    reference for that value as a matching-colour dashed h-line. ``label=False`` mutes
    legend entries (2nd panel of a broken-axis pair). Returns ``(fam_vals, cgs)``.
    """
    import matplotlib.pyplot as plt

    mesa = [r for r in rows if r["layer"] == "mesa" and r.get("cg_steps") is not None]
    gdn = [r for r in rows if r["layer"] == "gated_deltanet"]
    fam_vals = sorted({r[family] for r in mesa})
    cgs = sorted({r["cg_steps"] for r in mesa})
    colors = plt.get_cmap(cmap)(np.linspace(0.12, 0.88, max(len(fam_vals), 1)))
    for color, fv in zip(colors, fam_vals):
        pts = sorted((r for r in mesa if r[family] == fv), key=lambda r: r["cg_steps"])
        ax.plot([r["cg_steps"] for r in pts], [r["mse"] for r in pts], "o-",
                color=color, label=(f"{family}={fv}" if label else "_nolegend_"),
                lw=lw, ms=ms)
        if show_gdn:
            g = [r for r in gdn if r[family] == fv]
            if g:
                ax.axhline(g[0]["mse"], ls="--", lw=max(lw * 0.85, 0.7),
                           color="red", alpha=0.75)   # GDN = dashed red, everywhere
    return fam_vals, cgs


def _cg_xaxis(ax, cgs, logx):
    """Apply the shared CG-step x-axis (log ticks at the actual CG values)."""
    if logx:
        ax.set_xscale("log")
        ax.set_xticks(cgs)
        ax.set_xticklabels([str(c) for c in cgs])


def plot_cg_family(
    rows: list[dict],
    family: str,
    *,
    ax=None,
    title: str | None = None,
    cmap: str = "viridis",
    show_gdn: bool = True,
    logx: bool = True,
    logy: bool = False,
    lw: float = 1.0,
    ms: float = 4.0,
):
    """Held-out MSE vs Mesa CG steps, one solid curve per value of ``family``.

    ``family`` is 'noise' or 'drift'. Mesa rows (``cg_steps`` set) form the curves;
    if ``show_gdn`` and GDN rows are present, GDN is overlaid at each family value as a
    dashed line in the *matching* colour (its no-CG reference). Shows how the value of
    extra test-time compute (CG steps) changes as noise / drift grow. Returns the Axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.4))
    _, cgs = _draw_cg_family(ax, rows, family, cmap=cmap, show_gdn=show_gdn, lw=lw, ms=ms)
    _cg_xaxis(ax, cgs, logx)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("CG steps" + ("  (log)" if logx else ""))
    ax.set_ylabel("MSE" + ("  (log)" if logy else ""))
    if title:
        ax.set_title(title)
    if show_gdn and [r for r in rows if r["layer"] == "gated_deltanet"]:
        ax.plot([], [], "r--", label="Gated DeltaNet (dashed)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_cg_family_broken(
    rows: list[dict],
    family: str,
    *,
    lower_ylim: tuple[float, float],
    upper_ylim: tuple[float, float],
    height_ratios: tuple[float, float] = (1.0, 3.0),
    figsize: tuple[float, float] = (6.4, 4.8),
    title: str | None = None,
    cmap: str = "viridis",
    show_gdn: bool = True,
    logx: bool = True,
    lw: float = 0.9,
    ms: float = 4.0,
):
    """Broken (cut) y-axis version of :func:`plot_cg_family`.

    Same per-``family``-value CG-step curves, but split across two stacked axes, a
    small top panel (``upper_ylim``, the far-off high-MSE family value, e.g. noise=1.0)
    and a larger bottom panel (``lower_ylim``, the squeezed low-MSE families), with the
    empty band cut out and marked by diagonal break ticks. Keeps a linear y-scale on both
    so the low-MSE curves stop being crushed against the floor. Returns ``(top, bot)``.
    """
    import matplotlib.pyplot as plt

    fig, (top, bot) = plt.subplots(
        2, 1, sharex=True, figsize=figsize,
        gridspec_kw={"height_ratios": list(height_ratios), "hspace": 0.08},
    )
    _draw_cg_family(top, rows, family, cmap=cmap, show_gdn=show_gdn, lw=lw, ms=ms,
                    label=False)                                  # legend only on bottom
    _, cgs = _draw_cg_family(bot, rows, family, cmap=cmap, show_gdn=show_gdn, lw=lw, ms=ms)
    _cg_xaxis(bot, cgs, logx)

    top.set_ylim(*upper_ylim)
    bot.set_ylim(*lower_ylim)
    top.spines["bottom"].set_visible(False)
    bot.spines["top"].set_visible(False)
    top.tick_params(bottom=False, labelbottom=False)
    kw = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=8, linestyle="none",
              color="k", mec="k", mew=1, clip_on=False)
    top.plot([0, 1], [0, 0], transform=top.transAxes, **kw)
    bot.plot([0, 1], [1, 1], transform=bot.transAxes, **kw)

    for axis in (top, bot):
        axis.grid(True, which="both", alpha=0.3)
    bot.set_xlabel("CG steps" + ("  (log)" if logx else ""))
    bot.set_ylabel("MSE")
    bot.yaxis.set_label_coords(-0.1, 0.5 + height_ratios[0] / sum(height_ratios) / 2)
    if title:
        top.set_title(title)
    if show_gdn and [r for r in rows if r["layer"] == "gated_deltanet"]:
        bot.plot([], [], "r--", label="Gated DeltaNet (dashed)")
    bot.legend(fontsize=8)
    return top, bot


# --------------------------------------------------------------------------- #
# MQAR capacity sweep helpers (used by notebook 03)
#
# Design: TRAIN ONCE per (layer, CG) on a *distribution* mixing n_pairs AND gap,
# then EVALUATE that single frozen model at any (n_pairs, gap) setting. This is the
# train-across / evaluate-per-setting contract (cf. train_across_eval), extended to
# mix two difficulty axes at once, which the single-knob train_across_eval cannot do,
# hence this MQAR-specific trainer. The eval path reuses make_mqar + mqar_exact_match
# (the same scoring path as the data sanity check), so notebook 03 only orchestrates
# and plots.
# --------------------------------------------------------------------------- #
@dataclass
class MQARConfig:
    """Training/model config for the MQAR capacity experiment (notebook 03).

    ``train_pairs`` / ``train_gaps`` are the difficulty distributions a single model
    is trained across (each batch draws one (n_pairs, gap) pair uniformly); evaluation
    then probes any setting. ``vocab`` bounds the number of distinct keys (need
    ``vocab > max(train_pairs)``). The retention-init knobs match notebook 01/02's
    fairness convention (Mesa decay-gate bias; GDN initial per-step decay).
    """

    vocab: int = 128
    n_queries: int = 4
    train_pairs: tuple = (2, 4, 8, 16, 32, 64)
    train_gaps: tuple = (2, 8, 16, 32)
    train_distractors: tuple = (0,)     # distractor counts mixed in during training
    batch_size: int = 64
    steps: int = 2500
    lr: float = 2e-3
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0       # global-norm clip; tames Mesa's occasional CE blowups
    warmup: int = 200                   # linear LR warmup steps (stabilises early Mesa training)
    hidden_size: int = 128
    num_heads: int = 4
    num_layers: int = 2
    mlp_ratio: int = 4
    eval_batches: int = 8
    dtype: torch.dtype = torch.float32   # token path runs float32 (cf. compare.train)
    device: str | None = None
    mesa_retention_init: float | None = 4.0
    gdn_retention_init: float | None = None


class TrainedMQAR:
    """A single frozen model trained across the MQAR difficulty distribution.

    Holds the model plus its eval config; provides per-setting accuracy and a
    position-resolved (per-query-index) accuracy breakdown. Construct via
    :func:`train_mqar`. All eval batches use a disjoint seed range from training.
    """

    EVAL_SEED_BASE = 5_000_000

    def __init__(self, model, label, layer, cg_steps, mqcfg, device, train_loss=float("nan")):
        self.model = model
        self.label = label
        self.layer = layer
        self.cg_steps = cg_steps
        self.mqcfg = mqcfg
        self.device = device
        self.train_loss = train_loss
        self.num_params = count_params(model)

    @torch.no_grad()
    def _predict(self, inputs):
        x = torch.as_tensor(inputs, dtype=torch.long, device=self.device)
        return self.model(x).float().argmax(-1).cpu().numpy()  # (B, L) token ids

    def _eval_batches(self, n_pairs, gap, n_queries, eval_batches, seed, n_distractors=0):
        nq = n_queries or self.mqcfg.n_queries
        eb = eval_batches or self.mqcfg.eval_batches
        base = self.EVAL_SEED_BASE + seed
        for j in range(eb):
            yield make_mqar(self.mqcfg.batch_size, int(n_pairs), int(nq), int(gap),
                            self.mqcfg.vocab, base + j, n_distractors=int(n_distractors))

    @torch.no_grad()
    def accuracy(self, n_pairs, gap, *, n_queries=None, eval_batches=None, seed=0, n_distractors=0):
        """Held-out answer-token exact-match accuracy at one (n_pairs, gap, n_distractors) setting."""
        self.model.eval()
        accs = [mqar_exact_match(self._predict(inp), tgt, msk)
                for inp, tgt, msk in self._eval_batches(
                    n_pairs, gap, n_queries, eval_batches, seed, n_distractors)]
        return float(np.mean(accs))

    @torch.no_grad()
    def per_query_accuracy(self, n_pairs, gap, *, n_queries=None, eval_batches=None, seed=0,
                           n_distractors=0):
        """Accuracy at each query *index* (1st, 2nd, ... query token), depth probe.

        Later query indices sit deeper past the writes, so this reads off how recall
        holds up with sequence depth without a separate length sweep. Returns a list of
        length ``n_queries``.
        """
        self.model.eval()
        nq = n_queries or self.mqcfg.n_queries
        q_start = 2 * int(n_pairs) + 2 * int(n_distractors) + int(gap)
        sums = np.zeros(nq)
        nb = 0
        for inp, tgt, msk in self._eval_batches(n_pairs, gap, nq, eval_batches, seed, n_distractors):
            preds = self._predict(inp)
            for qi in range(nq):
                pos = q_start + qi
                sums[qi] += float((preds[:, pos] == tgt[:, pos]).mean())
            nb += 1
        return (sums / max(nb, 1)).tolist()


def _mqar_label(layer: str, cg_steps: int | None) -> str:
    """Curve label for an MQAR (layer, cg) condition (matches notebook 02 style)."""
    return _model_label(layer, cg_steps)


def train_mqar(layer: str, cg_steps: int | None, mqcfg: MQARConfig | None = None,
               seed: int = 0, use_cache: bool = True) -> TrainedMQAR:
    """Train ONE MQAR model across a mixed (n_pairs, gap) distribution; return it frozen.

    Each training batch draws ``n_pairs`` from ``mqcfg.train_pairs``, ``gap`` from
    ``mqcfg.train_gaps`` and ``n_distractors`` from ``mqcfg.train_distractors`` (uniform,
    independent), so the single model sees the whole capacity/retention/interference
    range. ``cg_steps`` applies to Mesa only. Loss is cross-entropy on the answer (masked)
    positions, the same scoring region the metric uses.

    With ``use_cache`` (default), weights for this exact config are reloaded from
    disk instead of retraining (see :mod:`compare.cache`); ``train_loss`` is NaN on
    a cache hit.
    """
    mqcfg = mqcfg or MQARConfig()
    dev = torch.device(mqcfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(seed)
    cfg = ModelConfig(
        layer=layer, input_kind="tokens", output_kind="classification",
        vocab=mqcfg.vocab, num_classes=mqcfg.vocab,
        hidden_size=mqcfg.hidden_size, num_heads=mqcfg.num_heads,
        num_layers=mqcfg.num_layers, mlp_ratio=mqcfg.mlp_ratio,
        cg_steps=(cg_steps if layer == "mesa" else None),
        mesa_retention_init=mqcfg.mesa_retention_init,
        gdn_retention_init=mqcfg.gdn_retention_init,
    )
    model = SequenceModel(cfg).to(dev).to(torch.float32)  # token path: float32

    # Weight-determining config -> disk-cache key. Eval-only mqcfg fields
    # (eval_batches) are excluded; training distributions and hyperparams are not.
    cache_spec = {
        "fn": "train_mqar",
        "layer": layer,
        "cg_steps": (cg_steps if layer == "mesa" else None),
        "seed": seed,
        "vocab": mqcfg.vocab,
        "n_queries": mqcfg.n_queries,
        "train_pairs": list(mqcfg.train_pairs),
        "train_gaps": list(mqcfg.train_gaps),
        "train_distractors": list(mqcfg.train_distractors),
        "batch_size": mqcfg.batch_size,
        "steps": mqcfg.steps,
        "lr": mqcfg.lr,
        "weight_decay": mqcfg.weight_decay,
        "grad_clip": mqcfg.grad_clip,
        "warmup": mqcfg.warmup,
        "hidden_size": mqcfg.hidden_size,
        "num_heads": mqcfg.num_heads,
        "num_layers": mqcfg.num_layers,
        "mlp_ratio": mqcfg.mlp_ratio,
        "dtype": str(mqcfg.dtype),
        "mesa_retention_init": mqcfg.mesa_retention_init,
        "gdn_retention_init": mqcfg.gdn_retention_init,
    }

    last_loss = float("nan")
    if not (use_cache and cache.try_load(model, cache_spec)):
        opt = torch.optim.AdamW(model.parameters(), lr=mqcfg.lr, weight_decay=mqcfg.weight_decay)
        warmup = max(int(mqcfg.warmup), 0)

        rng = np.random.default_rng(seed)
        pairs, gaps = list(mqcfg.train_pairs), list(mqcfg.train_gaps)
        dists = list(mqcfg.train_distractors)
        model.train()
        for step in range(mqcfg.steps):
            if warmup and step < warmup:                        # linear LR warmup
                for g in opt.param_groups:
                    g["lr"] = mqcfg.lr * (step + 1) / warmup
            n_pairs = int(pairs[rng.integers(len(pairs))])
            gap = int(gaps[rng.integers(len(gaps))])
            n_dist = int(dists[rng.integers(len(dists))])
            inputs, targets, mask = make_mqar(mqcfg.batch_size, n_pairs, mqcfg.n_queries,
                                              gap, mqcfg.vocab, seed + 1 + step,
                                              n_distractors=n_dist)
            x = torch.as_tensor(inputs, dtype=torch.long, device=dev)
            logits = model(x)                                   # (B, L, vocab)
            msk = torch.as_tensor(mask, dtype=torch.bool, device=dev)
            tgt = torch.as_tensor(targets, dtype=torch.long, device=dev)
            loss = F.cross_entropy(logits[msk], tgt[msk])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if mqcfg.grad_clip is not None:                     # tame occasional CE blowups
                torch.nn.utils.clip_grad_norm_(model.parameters(), mqcfg.grad_clip)
            opt.step()
            last_loss = float(loss.detach().cpu())
        if use_cache:
            cache.save(model, cache_spec)

    return TrainedMQAR(model, _mqar_label(layer, cg_steps), layer,
                       (cg_steps if layer == "mesa" else None), mqcfg, dev, train_loss=last_loss)


def train_mqar_models(specs: list[tuple], mqcfg: MQARConfig | None = None,
                      seed: int = 0, use_cache: bool = True) -> dict:
    """Train one :class:`TrainedMQAR` per ``(layer, cg_steps)`` spec; return by label.

    ``specs`` e.g. ``[("mesa", 1), ("mesa", 30), ("gated_deltanet", None)]``. The
    returned dict (keyed by curve label, insertion-ordered) is reused across every
    evaluation in notebook 03, train is the only expensive step. ``use_cache``
    reloads each model from disk when its config is unchanged (see
    :mod:`compare.cache`).
    """
    mqcfg = mqcfg or MQARConfig()
    out = {}
    for layer, cg in specs:
        tm = train_mqar(layer, cg, mqcfg, seed=seed, use_cache=use_cache)
        out[tm.label] = tm
    return out


def mqar_sweep_rows(models: dict, x: str, grid: list, *, n_pairs=None, gap=None,
                    n_distractors=0, n_queries=None, eval_batches=None, seed=0) -> list[dict]:
    """Evaluate each trained model over ``grid`` along axis ``x`` -> tidy accuracy rows.

    ``x`` is ``"n_pairs"``, ``"gap"`` or ``"n_distractors"``; the other two are held at the
    given ``n_pairs`` / ``gap`` / ``n_distractors`` values. Rows carry ``acc`` (answer-token
    exact-match) plus ``label / layer / cg_steps`` and every setting, ready for
    :func:`plot_mse_vs` with ``ykey="acc"``.
    """
    if x not in ("n_pairs", "gap", "n_distractors"):
        raise ValueError("x must be 'n_pairs', 'gap' or 'n_distractors'")
    rows = []
    for tm in models.values():
        for v in grid:
            np_ = v if x == "n_pairs" else n_pairs
            gp_ = v if x == "gap" else gap
            nd_ = v if x == "n_distractors" else n_distractors
            acc = tm.accuracy(np_, gp_, n_distractors=nd_, n_queries=n_queries,
                              eval_batches=eval_batches, seed=seed)
            rows.append({"label": tm.label, "layer": tm.layer, "cg_steps": tm.cg_steps,
                         x: v, "n_pairs": np_, "gap": gp_, "n_distractors": nd_,
                         "acc": acc, "num_params": tm.num_params})
    return rows


def _delta_to_mesa_bias(delta: float) -> float:
    """Target initial decay δ -> MesaNet a_proj.bias = logit(δ); δ→1 / δ→0 capped finite."""
    d = float(delta)
    if d >= 0.999:
        return 8.0       # σ(8) ≈ 0.9997, effectively no forgetting (logit(1) = ∞)
    if d <= 0.001:
        return -8.0
    return math.log(d / (1.0 - d))


def mqar_delta_cg_sweep(delta_grid: list, cg_grid: list, mqcfg: "MQARConfig", *,
                        n_pairs: int, gap: int, n_distractors: int, seed: int = 0,
                        with_gdn: bool = True) -> list[dict]:
    """δ × CG sweep on ONE fixed distractor-MQAR setting; one trained model per cell.

    For each δ in ``delta_grid``: trains a Mesa model at each CG in ``cg_grid`` (with
    ``mesa_retention_init = logit(δ)``) and, if ``with_gdn``, one GDN reference per δ
    (``gdn_retention_init = δ``, no CG dial). Every model is evaluated at the fixed
    ``(n_pairs, gap, n_distractors)``. The task/training distribution (vocab, n_queries,
    train_pairs/gaps/distractors, steps, ...) comes from ``mqcfg``; only the per-cell δ /
    CG are overridden, reusing :func:`train_mqar` and the matched-init path.

    Returns tidy rows, one per cell, with keys ``label, layer, cg_steps, delta,
    forget_rate, acc, num_params``. GDN rows carry ``cg_steps=None`` and ``layer=
    'gated_deltanet'``; feed the Mesa rows to :func:`plot_delta_cg_heatmap`.
    """
    rows: list[dict] = []
    for delta in delta_grid:
        mbias = _delta_to_mesa_bias(delta)
        for cg in cg_grid:
            tm = train_mqar("mesa", cg, replace(mqcfg, mesa_retention_init=mbias), seed=seed)
            acc = tm.accuracy(n_pairs, gap, n_distractors=n_distractors, seed=seed)
            rows.append({"label": _mqar_label("mesa", cg), "layer": "mesa", "cg_steps": cg,
                         "delta": float(delta), "forget_rate": round(1.0 - float(delta), 4),
                         "acc": acc, "num_params": tm.num_params})
        if with_gdn:
            gdn_delta = min(float(delta), 0.9997)   # gdn_retention_init must be in (0, 1)
            tm = train_mqar("gated_deltanet", None, replace(mqcfg, gdn_retention_init=gdn_delta),
                            seed=seed)
            acc = tm.accuracy(n_pairs, gap, n_distractors=n_distractors, seed=seed)
            rows.append({"label": "Gated DeltaNet", "layer": "gated_deltanet", "cg_steps": None,
                         "delta": float(delta), "forget_rate": round(1.0 - float(delta), 4),
                         "acc": acc, "num_params": tm.num_params})
    return rows


def plot_delta_cg_heatmap(rows: list[dict], *, ax=None, cmap: str = "viridis",
                          title: str | None = None, annotate: bool = True):
    """Headline heatmap: Mesa recall accuracy over (δ, CG); GDN best-δ noted in the title.

    Rows are from :func:`mqar_delta_cg_sweep`. y-axis = CG steps (ascending), x-axis =
    initial decay δ (retention increases rightward); each cell is answer-token accuracy
    in [0, 1]. The *shape* is the finding: flat-in-δ rows = separable levers; a tilt /
    curve = interaction (δ* shifts with CG). Returns the Axes.
    """
    import matplotlib.pyplot as plt

    mesa = [r for r in rows if r["layer"] == "mesa"]
    cgs = sorted({r["cg_steps"] for r in mesa})
    deltas = sorted({r["delta"] for r in mesa})
    grid = np.full((len(cgs), len(deltas)), np.nan)
    for r in mesa:
        grid[cgs.index(r["cg_steps"]), deltas.index(r["delta"])] = r["acc"]

    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 4.6))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(deltas)))
    ax.set_xticklabels([f"{d:g}" for d in deltas])
    ax.set_yticks(range(len(cgs)))
    ax.set_yticklabels([str(c) for c in cgs])
    ax.set_xlabel("δ (keep-rate)")
    ax.set_ylabel("CG steps")
    if annotate:
        for i in range(len(cgs)):
            for j in range(len(deltas)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                            color="white" if grid[i, j] < 0.55 else "black", fontsize=8)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("accuracy")
    t = title or "Mesa recall accuracy over (δ, CG) ,  distractor-MQAR"
    gdn = [r["acc"] for r in rows if r["layer"] == "gated_deltanet"]
    if gdn:
        t += f"\nGDN best-δ reference = {max(gdn):.2f}"
    ax.set_title(t)
    return ax


def mqar_cg_rows(models: dict, n_pairs, gap, *, n_queries=None, eval_batches=None,
                 seed=0) -> list[dict]:
    """Accuracy-vs-CG rows at one (n_pairs, gap): Mesa models on the CG axis, GDN as ref.

    Mesa rows get their ``cg_steps`` on the x-axis; GDN rows keep ``cg_steps=None`` so
    :func:`plot_mse_vs` draws them as a horizontal reference line (``ykey="acc"``).
    """
    rows = []
    for tm in models.values():
        acc = tm.accuracy(n_pairs, gap, n_queries=n_queries, eval_batches=eval_batches, seed=seed)
        rows.append({"label": tm.label, "layer": tm.layer, "cg_steps": tm.cg_steps,
                     "n_pairs": n_pairs, "gap": gap, "acc": acc, "num_params": tm.num_params})
    return rows


def mqar_position_rows(models: dict, n_pairs, gap, *, n_queries, eval_batches=None,
                       seed=0) -> list[dict]:
    """Per-query-index accuracy rows (depth probe) for each trained model.

    One row per (model, query index); ``query_pos`` is the 1-based query index. Feed to
    :func:`plot_mse_vs` with ``x="query_pos", ykey="acc"``.
    """
    rows = []
    for tm in models.values():
        accs = tm.per_query_accuracy(n_pairs, gap, n_queries=n_queries,
                                     eval_batches=eval_batches, seed=seed)
        for qi, a in enumerate(accs):
            rows.append({"label": tm.label, "layer": tm.layer, "cg_steps": tm.cg_steps,
                         "query_pos": qi + 1, "n_pairs": n_pairs, "gap": gap, "acc": a,
                         "num_params": tm.num_params})
    return rows


def plot_capacity_forget_control(rows_forget, rows_keep, *, x="n_pairs", ax=None,
                                 logx=True, ylabel="accuracy",
                                 xlabel=None, title=None, lw=1.3, ms=4.0):
    """Overlay the capacity sweep with forgetting OFF vs matched near-1 decay.

    ``rows_forget`` (decay δ→1, *solid*) and ``rows_keep`` (matched δ≈0.98, *dashed*) are
    accuracy rows from :func:`mqar_sweep_rows`; each model label gets one colour, the two
    decay settings differ only by linestyle. Disentangles the two causes of the high-load
    collapse: if the solid (no-forget) curves hold up to *higher* ``x`` than the dashed
    ones, the original collapse was partly *temporal forgetting* of early writes, not state
    capacity. Returns the Axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.6, 4.6))
    labels = []
    for r in list(rows_forget) + list(rows_keep):
        if r["label"] not in labels:
            labels.append(r["label"])
    cmap = plt.get_cmap("tab10")
    for i, lb in enumerate(labels):
        color = "red" if lb == "Gated DeltaNet" else cmap(i % 10)
        for rows, ls, suff in ((rows_forget, "-", "δ→1"),
                               (rows_keep, "--", "δ≈0.98")):
            pts = sorted((r for r in rows if r["label"] == lb and r.get(x) is not None),
                         key=lambda r: r[x])
            if not pts:
                continue
            ax.plot([r[x] for r in pts], [r["acc"] for r in pts], ls, marker="o",
                    color=color, lw=lw, ms=ms, label=f"{lb} {suff}")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    return ax


def mqar_flops_note(cg_steps: int) -> str:
    """One-line analytic FLOPs annotation: Mesa(CG=k) ≈ k× the GLA (CG=1) mixer cost.

    The CG solver runs one GLA-equivalent matmul pass per iteration (cold-start x=0; see
    the verified CG-semantics finding), so k iterations ≈ k× a single gated-linear-attn
    readout. GDN is a single delta pass ≈ 1× GLA. Mixer-only; ignores embed/MLP/head.
    """
    return f"Mesa(CG={cg_steps}) ≈ {cg_steps}× GLA mixer FLOPs  (GDN ≈ 1× GLA)"


def benchmark_mqar_latency(models: dict, n_pairs, gap, *, n_queries=None,
                           batch_size=None, reps: int = 30, warmup: int = 5,
                           seed: int = 0) -> dict:
    """Median forward-pass wall-clock (ms) per model at one fixed MQAR setting.

    Times ``model(x)`` only (no grad), on one shared input batch, with warmup and a
    CUDA sync after each timed call so the number reflects the real mixer cost rather
    than async launch latency. This is the *measured* counterpart to the analytic
    :func:`mqar_flops_note` — the latency ratios are smaller than the FLOPs ratios
    because embed/MLP/head overhead is shared across all conditions.

    Returns ``{label: median_ms}`` (insertion-ordered like ``models``).
    """
    import time

    any_tm = next(iter(models.values()))
    bs = int(batch_size or any_tm.mqcfg.batch_size)
    nq = int(n_queries or any_tm.mqcfg.n_queries)
    vocab = int(any_tm.mqcfg.vocab)
    inputs, _, _ = make_mqar(bs, int(n_pairs), nq, int(gap), vocab, 9_000_000 + seed)

    out = {}
    for lb, tm in models.items():
        dev = torch.device(tm.device)
        x = torch.as_tensor(inputs, dtype=torch.long, device=dev)
        tm.model.eval()
        times = []
        with torch.no_grad():
            for _ in range(warmup):
                tm.model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            for _ in range(reps):
                t0 = time.perf_counter()
                tm.model(x)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1e3)
        out[lb] = float(np.median(times))
    return out


def _pf(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


if __name__ == "__main__":
    print("=== Mesa collapse-control checks (CPU, no Triton) ===\n")
    cg = verify_cg_semantics()
    lam = check_lambda_collapse()

    print("[CG-step collapse]  Mesa(CG iterations) endpoints vs the GLA anchor")
    print(f"  CG=0  -> zero        ||o(CG=0)||            = {cg.cg0_output_norm:.2e}   "
          f"[{_pf(cg.pass_cg0_is_zero)}] CG=0 is zero (NOT GLA)")
    print(f"  CG=1  -> GLA         std/mean(o_CG1/o_gla)  = {cg.cg1_vs_gla_parallelism:.2e}   "
          f"[{_pf(cg.pass_cg1_is_gla)}] CG=1 == GLA up to a scalar")
    print(f"  CG=30 -> exact       ||o_CG30-o_exact||/||.|| = {cg.cg30_rel_error:.2e}   "
          f"[{_pf(cg.pass_converges_to_exact)}] converges to exact")

    print(f"\n[Λ collapse]        Mesa(Λ={lam.lambda_value:.0f}·I) vs the SAME GLA anchor, "
          f"up to a constant rescale α")
    print(f"  α (rescale)          = {lam.alpha:.4f}     (theory 1/Λ = {1.0 / lam.lambda_value:.4f})")
    print(f"  cosine(o_Λ, o_gla)   = {lam.cosine:.6f}   "
          f"[{_pf(lam.pass_directional)}] collapses to GLA direction")
    print(f"  rescaled query-MSE   = {lam.query_mse:.2e} (rel {lam.rel_mse:.2e})   "
          f"[{_pf(lam.pass_rescaled)}] residual << signal")
    print(f"  max/mean |α·o_gla-o_Λ| = {lam.max_abs_diff:.2e} / {lam.mean_abs_diff:.2e}")

    print("\n--- collapse-control summary ---")
    print(f"  {'CG-step collapse  (CG: 0->zero, 1->GLA, 30->exact)':52s} {_pf(cg.all_pass)}")
    print(f"  {'Λ collapse        (large Λ -> GLA, up to rescale)':52s} {_pf(lam.all_pass)}")
    print(f"\n  overall: {_pf(cg.all_pass and lam.all_pass)}")
