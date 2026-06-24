"""Disk cache for trained models: save weights keyed by the training config and
reload them instead of retraining. Enabled by default for every sweep the
notebooks run, so re-executing a notebook on the same machine skips the (slow)
training loops and re-runs only the cheap eval forward pass.

A cache entry is keyed by a SHA-256 over *everything that determines the trained
weights*: the layer, the task and its generator kwargs, the difficulty knob(s),
the seed, the optimisation hyperparams, the model shape, the compute dtype, and
the fairness/decay knobs. Eval-only settings (``eval_batches``, ``eval_knobs``,
``device``) are deliberately excluded — they do not change the weights, and a hit
re-runs only evaluation.

Controls (environment variables):
  MODEL_CACHE_OFF=1        disable the cache (always retrain, never read/write).
  MODEL_CACHE_DIR=/path    relocate the cache (default: <project_root>/.model_cache).

IMPORTANT: bump ``CACHE_VERSION`` whenever a change to the training code would
change the weights produced by an otherwise-unchanged config; otherwise stale
weights load silently. The version is part of every key, so bumping it cleanly
invalidates all existing entries.
"""

from __future__ import annotations

import hashlib
import json
import os

import torch

# Bump to invalidate every cached model after a training-code change.
CACHE_VERSION = 1

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", os.path.join(_ROOT, ".model_cache"))


def cache_enabled() -> bool:
    """False iff MODEL_CACHE_OFF is set to a truthy value."""
    return os.environ.get("MODEL_CACHE_OFF", "") not in ("1", "true", "True", "yes")


def _key(spec: dict) -> str:
    payload = {"v": CACHE_VERSION, **spec}
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def cache_path(spec: dict) -> str:
    """Absolute path of the cache file for ``spec`` (not guaranteed to exist)."""
    return os.path.join(CACHE_DIR, f"{_key(spec)}.pt")


def try_load(model: torch.nn.Module, spec: dict) -> bool:
    """Load cached weights into ``model`` in place; return True on a cache hit.

    No-op returning False when the cache is disabled or the entry is missing.
    Weights are mapped onto the model's current device.
    """
    if not cache_enabled():
        return False
    path = cache_path(spec)
    if not os.path.exists(path):
        return False
    device = next(model.parameters()).device
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return True


def save(model: torch.nn.Module, spec: dict) -> None:
    """Persist ``model``'s weights for ``spec`` via an atomic replace."""
    if not cache_enabled():
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = cache_path(spec)
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, path)
