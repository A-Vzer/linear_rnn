"""Compute-cost profiling for the swappable mixing layers.

The sweep notebooks put a *dial* on the x-axis (CG steps `k`, ridge `Λ`); a dial
is not compute. This module turns a model condition into the metrics that
actually matter for an LLM-style efficiency claim, so a quality curve can be
re-plotted against real cost (FLOPs / latency) instead of the dial:

  - **FLOPs / token** (mixer-only *and* whole-model) — analytical, deterministic,
    hardware-independent. The portable "compute axis." Counted from the layer
    math because fla's Triton kernels are invisible to ``torch`` FLOP counters.
  - **Throughput**, split into **prefill** (parallel chunk kernel) and **decode**
    (recurrent one-step kernel) tokens/s — the two regimes behave oppositely.
  - **Latency**: TTFT (prefill) and TPOT (decode inter-token), the LLM UX numbers.
  - **Peak memory** + **recurrent state size** — the linear-attention headline:
    state is O(1) in sequence length (vs attention's O(L) KV cache).
  - **Params** — fairness confound (are Mesa and GDN matched?).

FLOPs vs throughput are *not* the same thing: FLOPs is work per token (a property
of the model+shapes); throughput is tokens/s on this GPU (depends on kernel MFU
and memory bandwidth). For these layers the two diverge — that divergence is a
finding, not a bug — so we report both.

Convention: 1 multiply-accumulate = 2 FLOPs. Counts are leading-order (dense
matmuls in the projections, MLP, and the token-mixing core); RMSNorms, biases,
elementwise gates, and the short convolutions are O(H) or O(H·conv) and dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import ModelConfig, SequenceModel, count_params

__all__ = ["ProfileResult", "compute_profile", "model_flops_per_token", "mixer_flops_per_token"]


# --------------------------------------------------------------------------- #
# Analytical FLOPs (no GPU needed; pure function of shapes + cg_steps)
# --------------------------------------------------------------------------- #
def _matmul_flops(d_in: int, d_out: int) -> int:
    """FLOPs for one token through a (d_in -> d_out) dense projection (MAC = 2)."""
    return 2 * d_in * d_out


def mixer_flops_per_token(layer: str, hidden_size: int, num_heads: int,
                          cg_steps: int | None = None) -> int:
    """Leading-order FLOPs/token for ONE mixing layer (projections + token-mixing core).

    The projections (q, k, v, output-gate, o_proj — all H<->H here) are identical
    across Mesa and GDN; the *core* is the differentiator:

      - MesaNet: rank-1 updates to the two d_h x d_h states + ``cg_steps`` conjugate-
        gradient matvecs against the d_h x d_h key-correlation state + a readout, so
        the core is ``2·H²/heads·(cg_steps + 3)`` — **linear in cg_steps**.
      - GatedDeltaNet: one delta-rule update (state matvec, rank-1 write, decay,
        readout), ``≈ 2·H²/heads·4`` — **constant** in any dial.
    """
    H, h = hidden_size, num_heads
    d_h = H // h
    # shared linear projections: q, k, v, output gate, o_proj (each H<->H)
    proj = 5 * _matmul_flops(H, H)
    if layer == "mesa":
        m = 30 if cg_steps is None else int(cg_steps)
        core = h * (2 * d_h * d_h) * (m + 3)   # 2 state updates + m CG matvecs + readout
    elif layer == "gated_deltanet":
        core = h * (2 * d_h * d_h) * 4         # matvec + rank-1 write + decay + readout
    elif layer == "mock":
        core = 0
    else:
        raise ValueError(f"unknown layer {layer!r}")
    return proj + core


def model_flops_per_token(cfg: ModelConfig, cg_steps: int | None = None) -> tuple[int, int]:
    """(whole-model, mixer-only) FLOPs/token for the full :class:`SequenceModel`.

    Whole-model = in-proj + num_layers·(mixer + MLP) + head; mixer-only sums just
    the token-mixing layers (the part that moves with the compute dial).
    """
    H = cfg.hidden_size
    in_dim = cfg.input_dim if cfg.input_dim is not None else H
    f_in = _matmul_flops(in_dim, H)
    f_mlp = 2 * _matmul_flops(H, cfg.mlp_ratio * H)                 # fc1 + fc2
    f_mix = mixer_flops_per_token(cfg.layer, H, cfg.num_heads, cg_steps)
    f_head = _matmul_flops(H, cfg.num_classes if cfg.num_classes else 1)
    mixer_total = cfg.num_layers * f_mix
    whole = f_in + cfg.num_layers * (f_mix + f_mlp) + f_head
    return whole, mixer_total


def recurrent_state_bytes(cfg: ModelConfig, dtype: torch.dtype, conv_size: int = 4) -> int:
    """Bytes of recurrent state carried between tokens — **constant in seq length**.

    Mesa keeps two d_h x d_h states per head (KᵀK and KᵀV); GDN keeps one. The
    short-conv state (conv_size-1 per channel, q & k) is tiny but included.
    """
    H, h = cfg.hidden_size, cfg.num_heads
    d_h = H // h
    n_state = {"mesa": 2, "gated_deltanet": 1, "mock": 0}.get(cfg.layer, 0)
    elt = torch.empty(0, dtype=dtype).element_size()
    recur = cfg.num_layers * h * (d_h * d_h) * n_state * elt
    conv = cfg.num_layers * 2 * H * (conv_size - 1) * elt
    return recur + conv


# --------------------------------------------------------------------------- #
# Measured throughput / latency / memory (CUDA)
# --------------------------------------------------------------------------- #
def _bench(fn, warmup: int, iters: int) -> float:
    """Median wall-clock (ms) of ``fn`` over ``iters`` runs after ``warmup`` (CUDA-event timed)."""
    for _ in range(warmup):       # Triton autotune compiles on first calls — must warm up
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


@dataclass
class ProfileResult:
    """Compute profile of one model condition (see :func:`compute_profile`)."""

    layer: str
    cg_steps: int | None
    params: int
    flops_per_token_total: int        # whole model, analytical
    flops_per_token_mixer: int        # token-mixing layers only
    state_bytes: int                  # recurrent state, constant in seq length
    prefill_tok_s: float              # measured; nan if timing skipped
    decode_tok_s: float
    ttft_ms: float                    # prefill latency (time-to-first-token)
    tpot_ms: float                    # decode inter-token latency
    peak_mem_mb: float

    def row(self) -> str:
        tag = self.layer + (f" (CG={self.cg_steps})" if self.cg_steps is not None else "")
        return (f"{tag:18s} params={self.params/1e3:6.1f}k  "
                f"FLOPs/tok: total={self.flops_per_token_total/1e3:7.1f}k "
                f"mixer={self.flops_per_token_mixer/1e3:6.1f}k  "
                f"state={self.state_bytes/1024:6.1f}KiB  "
                f"prefill={self.prefill_tok_s/1e3:7.1f}k tok/s  "
                f"decode={self.decode_tok_s:6.0f} tok/s (TPOT {self.tpot_ms:.2f}ms)  "
                f"peak={self.peak_mem_mb:5.0f}MB")


def compute_profile(
    layer: str,
    cfg,
    *,
    cg_steps: int | None = None,
    seq_len: int = 128,            # eval shape: 2 * max(EVAL_EXAMPLES) for regression
    batch: int = 64,
    decode_batch: int = 1,
    input_kind: str = "continuous",
    input_dim: int | None = None,  # default cfg.d + 1 (regression token dim)
    vocab: int | None = None,
    output_kind: str = "regression",
    num_classes: int = 1,
    warmup: int = 5,
    iters: int = 20,
    measure_timing: bool = True,
) -> ProfileResult:
    """Profile one ``(layer, cg_steps)`` condition built from a ``SweepConfig``-like ``cfg``.

    Analytical metrics (FLOPs, state bytes, params) are always returned. Throughput,
    latency, and peak memory are measured on CUDA when ``measure_timing`` is True and
    ``cfg.device == 'cuda'`` (else returned as nan). Defaults match notebook 01's eval
    shape: prefill ``(batch=64, seq_len=128)`` and decode ``batch=1``. Timing depends
    only on tensor *shapes*, so the regression I/O here is a valid stand-in for any
    continuous task; pass ``input_kind='tokens'`` (+ ``vocab``, ``num_classes``) for
    token tasks like MQAR.
    """
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = cfg.dtype if input_kind == "continuous" else torch.float32
    if input_dim is None:
        input_dim = cfg.d + 1
    mcfg = ModelConfig(
        layer=layer, input_kind=input_kind, output_kind=output_kind,
        input_dim=(input_dim if input_kind == "continuous" else None),
        vocab=(vocab if input_kind == "tokens" else None),
        num_classes=num_classes,
        hidden_size=cfg.hidden_size, num_heads=cfg.num_heads, num_layers=cfg.num_layers,
        cg_steps=cg_steps,
        mesa_retention_init=cfg.mesa_retention_init,
        gdn_retention_init=cfg.gdn_retention_init,
    )
    whole, mixer = model_flops_per_token(mcfg, cg_steps)
    state_bytes = recurrent_state_bytes(mcfg, dtype)
    model = SequenceModel(mcfg).to(device).to(dtype if input_kind == "continuous" else torch.float32)
    model.eval()
    params = count_params(model)

    nan = float("nan")
    prefill_tok_s = decode_tok_s = ttft_ms = tpot_ms = peak_mem_mb = nan
    if measure_timing and device.type == "cuda":
        from fla.models.utils import Cache

        def make_x(b: int, t: int):
            if input_kind == "continuous":
                return torch.randn(b, t, input_dim, device=device, dtype=dtype)
            return torch.randint(0, vocab or 2, (b, t), device=device)

        x_pre = make_x(batch, seq_len)

        # --- prefill: parallel chunk kernel over the whole prompt ---
        @torch.no_grad()
        def prefill():
            model(x_pre)
        ttft_ms = _bench(prefill, warmup, iters)
        prefill_tok_s = batch * seq_len * 1e3 / ttft_ms

        # --- peak memory of a prefill forward ---
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            model(x_pre)
        torch.cuda.synchronize()
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6

        # --- decode: warm a cache with a prompt, then time single recurrent steps ---
        cache = Cache()
        x_ctx, x_one = make_x(decode_batch, seq_len), make_x(decode_batch, 1)
        with torch.no_grad():
            model(x_ctx, past_key_values=cache, use_cache=True)

        @torch.no_grad()
        def step():
            model(x_one, past_key_values=cache, use_cache=True)
        tpot_ms = _bench(step, warmup, iters)
        decode_tok_s = decode_batch * 1e3 / tpot_ms

    return ProfileResult(
        layer=layer, cg_steps=cg_steps, params=params,
        flops_per_token_total=whole, flops_per_token_mixer=mixer,
        state_bytes=state_bytes,
        prefill_tok_s=prefill_tok_s, decode_tok_s=decode_tok_s,
        ttft_ms=ttft_ms, tpot_ms=tpot_ms, peak_mem_mb=peak_mem_mb,
    )
