"""Minimal, reproducible, device-agnostic train/eval harness for the comparison.

``train_eval(layer, task_fn, knob_value, cg_steps=None, seed=..., **train_cfg)``
trains a tiny :class:`compare.model.SequenceModel` on fresh batches from a
synthtasks generator and returns the task-appropriate metric on held-out batches.

Crucially, scoring goes through ``synthtasks.metrics`` exactly as the closed-form
sanity check does, so the model's eval path is identical to the reference path.

``task_fn`` is the actual synthtasks generator (e.g. ``make_regression``); the
per-task wiring (which arg is the difficulty knob, the io kind, the metric, and
the loss) lives in :data:`TASK_SPECS`, keyed by the generator function itself.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

# Make the project root importable whether run as a module or a script.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from compare import cache  # noqa: E402
from compare.model import ModelConfig, SequenceModel, count_params  # noqa: E402
from synthtasks import metrics  # noqa: E402
from synthtasks.regression import make_regression  # noqa: E402
from synthtasks.mqar import make_mqar  # noqa: E402
from synthtasks.parity import make_parity  # noqa: E402


@dataclass(frozen=True)
class TaskSpec:
    """How to drive a synthtasks generator + score it like the sanity check.

    Attributes:
        knob: generator kwarg filled by ``knob_value`` (the difficulty dial).
        defaults: other generator kwargs (overridable via ``task_kwargs``).
        input_kind: "continuous" | "tokens".
        output_kind: "regression" | "classification".
        loss_kind: "mse" | "ce".
        metric: a ``synthtasks.metrics`` function (predictions, targets, mask).
        num_classes: maps resolved generator kwargs -> #classes (None=regression).
    """

    knob: str
    input_kind: str
    output_kind: str
    loss_kind: str
    metric: Callable[..., float]
    num_classes: Callable[[dict], int | None]
    defaults: dict = field(default_factory=dict)


# Registry keyed by the generator function so train_eval(layer, make_regression, ...)
# reads literally as "train on this generator". Defaults are sensible starting
# points; override any of them per-call via train_cfg["task_kwargs"].
TASK_SPECS: dict[Callable, TaskSpec] = {
    make_regression: TaskSpec(
        knob="n_examples",
        defaults={"d": 8, "noise": 0.0, "drift": 0.0},
        input_kind="continuous",
        output_kind="regression",
        loss_kind="mse",
        metric=metrics.mse_on_queries,
        num_classes=lambda kw: None,
    ),
    make_mqar: TaskSpec(
        knob="n_pairs",
        defaults={"n_queries": 4, "gap": 8, "vocab": 64},
        input_kind="tokens",
        output_kind="classification",
        loss_kind="ce",
        metric=metrics.mqar_exact_match,
        num_classes=lambda kw: int(kw["vocab"]),
    ),
    make_parity: TaskSpec(
        knob="seq_len",
        defaults={},
        input_kind="tokens",
        output_kind="classification",
        loss_kind="ce",
        metric=metrics.parity_per_position_accuracy,
        num_classes=lambda kw: 2,
    ),
}


def set_seed(seed: int) -> None:
    """Seed torch (model init) reproducibly; data is seeded via the generator."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_tensors(
    inputs: np.ndarray,
    input_kind: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Move generator inputs to the right tensor dtype/device for the model."""
    if input_kind == "continuous":
        return torch.as_tensor(inputs, dtype=dtype, device=device)
    return torch.as_tensor(inputs, dtype=torch.long, device=device)  # token ids


def _masked_loss(
    pred: torch.Tensor,
    targets: np.ndarray,
    mask: np.ndarray,
    loss_kind: str,
    device: torch.device,
) -> torch.Tensor:
    """Loss over masked positions only (matches the scored region)."""
    msk = torch.as_tensor(mask, dtype=torch.bool, device=device)
    if loss_kind == "mse":
        tgt = torch.as_tensor(targets, dtype=pred.dtype, device=device)
        diff = (pred - tgt)[msk]
        return (diff ** 2).mean()
    # cross-entropy on the masked answer positions
    tgt = torch.as_tensor(targets, dtype=torch.long, device=device)
    logits = pred[msk]            # (n_scored, num_classes)
    labels = tgt[msk]             # (n_scored,)
    return F.cross_entropy(logits, labels)


@dataclass
class EvalResult:
    """Returned when ``return_details=True``."""

    metric: float
    train_loss: float
    layer: str
    knob: str
    knob_value: int
    cg_steps: int | None
    num_params: int
    device: str


def train_eval(
    layer: str,
    task_fn: Callable,
    knob_value: int,
    cg_steps: int | None = None,
    seed: int = 0,
    *,
    batch_size: int = 64,
    steps: int = 300,
    lr: float = 3e-3,
    weight_decay: float = 0.0,
    hidden_size: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    mlp_ratio: int = 4,
    eval_batches: int = 8,
    device: str | None = None,
    dtype: torch.dtype = torch.float32,
    mesa_retention_init: float | None = None,
    gdn_retention_init: float | None = None,
    mesa_lambda: float | None = None,
    allow_neg_eigval: bool = False,
    task_kwargs: dict | None = None,
    return_details: bool = False,
    use_cache: bool = True,
):
    """Train a tiny model on a synthtasks generator; return the held-out metric.

    Args:
        layer: "mesa" | "gated_deltanet" | "mock".
        task_fn: a synthtasks generator (key of :data:`TASK_SPECS`).
        knob_value: difficulty value placed into the task's knob arg.
        cg_steps: Mesa CG steps (the sweep dial; ignored by other layers).
        seed: reproducibility seed for model init and (offset) data batches.
        batch_size, steps, lr, weight_decay: training-loop config.
        hidden_size, num_heads, num_layers, mlp_ratio: model config (tiny regime).
        eval_batches: number of held-out batches to average the metric over.
        device: "cuda"/"cpu"/... ; defaults to cuda if available else cpu.
        dtype: compute dtype. NOTE: fla chunk kernels typically want
            torch.bfloat16 on CUDA; float32 is fine for the mock/CPU path.
        mesa_retention_init: Mesa decay-gate bias init (see ModelConfig); equalises
            Mesa's forgetting prior with GDN's retentive one. Mesa-only.
        gdn_retention_init: GDN target initial per-step decay in (0, 1) (see
            ModelConfig); pins GDN's forgetting so it can be matched to Mesa. GDN-only.
        task_kwargs: overrides merged over the spec's generator defaults.
        return_details: if True, return an :class:`EvalResult` instead of float.
        use_cache: if True (default), reuse weights saved to disk for this exact
            config instead of retraining (see :mod:`compare.cache`); only the eval
            forward pass runs on a hit. ``train_loss`` is NaN on a cache hit.

    Returns:
        The held-out metric (float), or an :class:`EvalResult` if requested.
        For regression that metric is MSE (lower better); for mqar/parity it is
        accuracy (higher better) — exactly the synthtasks.metrics convention.
    """
    if task_fn not in TASK_SPECS:
        raise KeyError(f"no TaskSpec registered for {getattr(task_fn, '__name__', task_fn)!r}")
    spec = TASK_SPECS[task_fn]
    gen_kwargs = {**spec.defaults, **(task_kwargs or {})}

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(seed)

    def gen(batch: int, data_seed: int):
        return task_fn(batch=batch, seed=data_seed, **{spec.knob: knob_value}, **gen_kwargs)

    # Infer io shapes from a probe batch (keeps the harness task-agnostic).
    probe_inputs, _, _ = gen(2, seed)
    num_classes = spec.num_classes(gen_kwargs)
    cfg = ModelConfig(
        layer=layer,
        input_kind=spec.input_kind,
        output_kind=spec.output_kind,
        input_dim=(probe_inputs.shape[-1] if spec.input_kind == "continuous" else None),
        vocab=(num_classes if spec.input_kind == "tokens" else None),
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        cg_steps=cg_steps,
        mesa_retention_init=mesa_retention_init,
        gdn_retention_init=gdn_retention_init,
        mesa_lambda=mesa_lambda,
        allow_neg_eigval=allow_neg_eigval,
        mlp_ratio=mlp_ratio,
    )
    model = SequenceModel(cfg).to(dev).to(dtype if spec.input_kind == "continuous" else torch.float32)

    # Everything that determines the trained weights -> the disk-cache key.
    cache_spec = {
        "fn": "train_eval",
        "task": getattr(task_fn, "__name__", str(task_fn)),
        "gen_kwargs": gen_kwargs,
        "knob": spec.knob,
        "knob_value": knob_value,
        "layer": layer,
        "cg_steps": cg_steps,
        "seed": seed,
        "batch_size": batch_size,
        "steps": steps,
        "lr": lr,
        "weight_decay": weight_decay,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "mlp_ratio": mlp_ratio,
        "dtype": str(dtype),
        "mesa_retention_init": mesa_retention_init,
        "gdn_retention_init": gdn_retention_init,
        "mesa_lambda": mesa_lambda,
        "allow_neg_eigval": allow_neg_eigval,
    }

    # --- train on fresh random batches (amortized in-context learning) ---
    # unless cached weights for this exact config exist, in which case reload them.
    last_loss = float("nan")
    if not (use_cache and cache.try_load(model, cache_spec)):
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        model.train()
        for step in range(steps):
            inputs, targets, mask = gen(batch_size, seed + 1 + step)  # training seeds
            x = _to_tensors(inputs, spec.input_kind, dev, dtype)
            pred = model(x)
            loss = _masked_loss(pred, targets, mask, spec.loss_kind, dev)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu())
        if use_cache:
            cache.save(model, cache_spec)

    # --- eval on held-out batches (disjoint seed range) via synthtasks.metrics ---
    model.eval()
    scores = []
    with torch.no_grad():
        for j in range(eval_batches):
            inputs, targets, mask = gen(batch_size, seed + 1_000_000 + j)  # held-out
            x = _to_tensors(inputs, spec.input_kind, dev, dtype)
            pred = model(x).float().cpu().numpy()
            scores.append(spec.metric(pred, targets, mask))  # identical scoring path
    metric_value = float(np.mean(scores))

    if return_details:
        return EvalResult(
            metric=metric_value,
            train_loss=last_loss,
            layer=layer,
            knob=spec.knob,
            knob_value=knob_value,
            cg_steps=cg_steps,
            num_params=count_params(model),
            device=str(dev),
        )
    return metric_value


def train_across_eval(
    layer: str,
    task_fn: Callable,
    train_knobs: list[int],
    eval_knobs: list[int],
    cg_steps: int | None = None,
    seed: int = 0,
    *,
    batch_size: int = 64,
    steps: int = 400,
    lr: float = 3e-3,
    weight_decay: float = 0.0,
    hidden_size: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    mlp_ratio: int = 4,
    eval_batches: int = 8,
    device: str | None = None,
    dtype: torch.dtype = torch.float32,
    mesa_retention_init: float | None = None,
    gdn_retention_init: float | None = None,
    mesa_lambda: float | None = None,
    allow_neg_eigval: bool = False,
    task_kwargs: dict | None = None,
    use_cache: bool = True,
) -> dict:
    """Train ONE model across a distribution of the knob; evaluate per setting.

    This is the default experimental design: each training batch draws its knob
    value (e.g. n_examples) uniformly from ``train_knobs``, so a single model
    sees the whole difficulty range; the held-out metric is then measured
    separately at each value in ``eval_knobs``. (Each batch is homogeneous in
    length — only across batches does the knob vary.) Contrast with
    :func:`train_eval`, which trains a fresh model per knob value.

    Args:
        layer, task_fn, cg_steps, seed: as in :func:`train_eval`.
        train_knobs: knob values sampled (uniformly, per batch) during training.
        eval_knobs: knob values at which to report held-out metrics.
        use_cache: reuse disk-cached weights for this exact config instead of
            retraining (see :mod:`compare.cache`); ``train_loss`` is NaN on a hit.
        (remaining kwargs): as in :func:`train_eval`.

    Returns:
        Dict with ``eval_knobs`` and the matching ``metric`` list, plus
        ``train_loss``, ``layer``, ``knob`` (name), ``cg_steps``, ``num_params``.
        Scoring uses ``synthtasks.metrics`` — identical to the sanity check.
    """
    if task_fn not in TASK_SPECS:
        raise KeyError(f"no TaskSpec registered for {getattr(task_fn, '__name__', task_fn)!r}")
    spec = TASK_SPECS[task_fn]
    gen_kwargs = {**spec.defaults, **(task_kwargs or {})}
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(seed)

    def gen(batch: int, knob: int, data_seed: int):
        return task_fn(batch=batch, seed=data_seed, **{spec.knob: knob}, **gen_kwargs)

    # io shapes are independent of the knob; probe at the largest value to be safe.
    probe_inputs, _, _ = gen(2, max(eval_knobs + train_knobs), seed)
    num_classes = spec.num_classes(gen_kwargs)
    cfg = ModelConfig(
        layer=layer,
        input_kind=spec.input_kind,
        output_kind=spec.output_kind,
        input_dim=(probe_inputs.shape[-1] if spec.input_kind == "continuous" else None),
        vocab=(num_classes if spec.input_kind == "tokens" else None),
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        cg_steps=cg_steps,
        mesa_retention_init=mesa_retention_init,
        gdn_retention_init=gdn_retention_init,
        mesa_lambda=mesa_lambda,
        allow_neg_eigval=allow_neg_eigval,
        mlp_ratio=mlp_ratio,
    )
    compute_dtype = dtype if spec.input_kind == "continuous" else torch.float32
    model = SequenceModel(cfg).to(dev).to(compute_dtype)

    # Weight-determining config -> disk-cache key. ``eval_knobs`` is excluded: it
    # does not affect the trained weights, only what we measure afterwards.
    cache_spec = {
        "fn": "train_across_eval",
        "task": getattr(task_fn, "__name__", str(task_fn)),
        "gen_kwargs": gen_kwargs,
        "knob": spec.knob,
        "train_knobs": sorted(int(k) for k in train_knobs),
        "layer": layer,
        "cg_steps": cg_steps,
        "seed": seed,
        "batch_size": batch_size,
        "steps": steps,
        "lr": lr,
        "weight_decay": weight_decay,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "mlp_ratio": mlp_ratio,
        "dtype": str(dtype),
        "mesa_retention_init": mesa_retention_init,
        "gdn_retention_init": gdn_retention_init,
        "mesa_lambda": mesa_lambda,
        "allow_neg_eigval": allow_neg_eigval,
    }

    # --- train across the knob distribution (unless cached weights exist) ---
    last_loss = float("nan")
    if not (use_cache and cache.try_load(model, cache_spec)):
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        knob_rng = np.random.default_rng(seed)
        model.train()
        for step in range(steps):
            knob = int(train_knobs[knob_rng.integers(len(train_knobs))])
            inputs, targets, mask = gen(batch_size, knob, seed + 1 + step)
            x = _to_tensors(inputs, spec.input_kind, dev, dtype)
            loss = _masked_loss(model(x), targets, mask, spec.loss_kind, dev)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu())
        if use_cache:
            cache.save(model, cache_spec)

    # --- evaluate per setting on held-out (disjoint-seed) batches ---
    model.eval()
    per_knob = []
    with torch.no_grad():
        for e, knob in enumerate(eval_knobs):
            scores = []
            for j in range(eval_batches):
                inputs, targets, mask = gen(batch_size, knob, seed + 5_000_000 + e * 10_000 + j)
                pred = model(_to_tensors(inputs, spec.input_kind, dev, dtype)).float().cpu().numpy()
                scores.append(spec.metric(pred, targets, mask))
            per_knob.append(float(np.mean(scores)))

    return {
        "eval_knobs": list(eval_knobs),
        "metric": per_knob,
        "train_loss": last_loss,
        "layer": layer,
        "knob": spec.knob,
        "cg_steps": cg_steps,
        "num_params": count_params(model),
        "device": str(dev),
    }


def _fla_available() -> bool:
    try:
        import triton  # noqa: F401
        import fla  # noqa: F401
        return True
    except Exception:
        return False


if __name__ == "__main__":
    # Smoke test: the MOCK mixer must drive in-context-regression MSE well below
    # the ~1.0 naive variance, proving the train/eval/masking/metric path works.
    print("=== train.py smoke test ===")
    print(f"fla+triton available: {_fla_available()}  "
          f"(mesa/gated_deltanet need this; mock does not)\n")

    res = train_eval(
        layer="mock",
        task_fn=make_regression,
        knob_value=16,              # n_examples
        seed=0,
        steps=300,
        batch_size=128,
        hidden_size=64,
        num_heads=4,
        task_kwargs={"d": 4, "noise": 0.0},
        return_details=True,
    )
    print(f"[mock] regression  n_examples=16  d=4")
    print(f"  params={res.num_params:,}  device={res.device}  "
          f"final_train_loss={res.train_loss:.4f}")
    print(f"  held-out MSE = {res.metric:.4f}   "
          f"(naive variance ~1.0; << 1.0 means the harness learns)")

    if _fla_available() and torch.cuda.is_available():
        for layer in ("mesa", "gated_deltanet"):
            cg = 5 if layer == "mesa" else None
            r = train_eval(layer, make_regression, 16, cg_steps=cg, seed=0,
                           steps=300, batch_size=128, hidden_size=64, num_heads=4,
                           dtype=torch.bfloat16, task_kwargs={"d": 4},
                           return_details=True)
            print(f"[{layer}] held-out MSE = {r.metric:.4f}  cg_steps={r.cg_steps}")
    else:
        print("\n[skip] mesa/gated_deltanet require CUDA+Triton; run the notebook "
              "on a GPU box for the real comparison.")
