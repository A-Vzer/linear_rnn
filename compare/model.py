"""Minimal sequence model wrapping a single swappable fla token-mixing layer.

Design goal: the *only* thing that varies between conditions is the mixing layer.
Everything else — the in-projection (Linear for continuous inputs / Embedding for
token inputs), the pre-norm residual block structure, the MLP, the RMSNorm
normalization, and the per-head output gating — is identical for both layers.

Conditions (``ModelConfig.layer``):
  "mesa"          -> fla.layers.mesa_net.MesaNet      (CG steps = sweep dial)
  "gated_deltanet"-> fla.layers.gated_deltanet.GatedDeltaNet
  "mock"          -> pure-torch causal attention; CPU/Mac smoke-test ONLY,
                     NOT a comparison condition (fla needs Triton/GPU).

fla API verified against installed fla-core 0.5.1 by reading the source:
  .venv/.../fla/layers/mesa_net.py
      class MesaNet(hidden_size, num_heads, head_dim, mode='chunk',
                    use_output_gate=False, use_short_conv=True, ...,
                    max_cg_step_training=30, max_cg_step_decoding=30)
      -> the CG-iteration count is max_cg_step_{training,decoding}; this IS the dial.
  .venv/.../fla/layers/gated_deltanet.py
      class GatedDeltaNet(hidden_size, expand_v=2, head_dim=256, num_heads=6,
                          num_v_heads=None, mode='chunk', use_gate=True,
                          use_short_conv=True, allow_neg_eigval=False, ...)
  Both .forward(hidden_states, attention_mask=None, ...) -> (o, None, past_kv);
  both require mode='chunk' for training. NOTE: their ``attention_mask`` is a
  padding mask, NOT our scoring mask — we never pass the synthtasks mask here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# fla layers are imported LAZILY inside build_mixing_layer so this module imports
# fine on machines without Triton (e.g. macOS); only the "mock" path runs there.


@dataclass
class ModelConfig:
    """Configuration for :class:`SequenceModel`.

    Attributes:
        layer: "mesa" | "gated_deltanet" | "delta_net" | "mock".
        input_kind: "continuous" (Linear in-proj) | "tokens" (Embedding).
        output_kind: "regression" (scalar/pos) | "classification" (logits/pos).
        input_dim: feature dim for continuous inputs (token-dim of the sequence).
        vocab: vocabulary size for token inputs (Embedding rows).
        num_classes: output classes for classification.
        hidden_size: model width; must be divisible by num_heads.
        num_heads: attention/mixer heads (head_dim = hidden_size // num_heads).
        num_layers: number of residual blocks (tiny regime: 2).
        cg_steps: Mesa conjugate-gradient steps (ignored for other layers).
        mesa_retention_init: if set, initialise MesaNet's decay-gate bias
            (``a_proj.bias``) to this value so the per-step state decay starts near
            ``exp(logsigmoid(value))`` ≈ 1 (retain history) instead of fla's stock
            ≈ 0.5/step (forget fast). Equalises Mesa's init prior with GDN's
            Mamba-style near-no-forgetting init; ignored for other layers. None =
            stock fla init. (See build_mixing_layer for the why.)
        gdn_retention_init: if set, force Gated DeltaNet's initial per-step state
            decay to this value in (0, 1) — 1 = retain everything — by setting
            ``A_log=0`` and ``dt_bias`` accordingly. fla's stock GDN init gives a
            *heterogeneous* per-head decay (~0.2-1.0); this pins it to one near-1
            value so its forgetting can be matched to Mesa's. None = stock fla init.
        mesa_lambda: if set, pin MesaNet's ridge regularizer Λ to this value and
            freeze it, so it can be swept as the regularization strength (larger Λ =
            more regularized -> degraded, GLA-ward solve). None = trainable Λ
            (stock). Mesa-only. (See build_mixing_layer.)
        allow_neg_eigval: widen the state-transition eigenvalues to γ ∈ (−1, 1) so the
            layer can express the bit-flip (eigenvalue −1) that state-tracking (parity)
            needs. GDN-only: maps to fla GatedDeltaNet's ``allow_neg_eigval`` (β is
            scaled ×2; Grazzi/Sarrof). **Not available for MesaNet** — its decay gate is
            ``logsigmoid`` (so γ ∈ (0,1)) fed to a log-cumsum chunk kernel that cannot
            represent γ<0, and the solve needs (H_t+Λ) positive-definite; setting this on
            "mesa" raises. Default False = stock (0,1) gate.
        mlp_ratio: MLP hidden expansion factor.
    """

    layer: str
    input_kind: str
    output_kind: str
    input_dim: int | None = None
    vocab: int | None = None
    num_classes: int | None = None
    hidden_size: int = 128
    num_heads: int = 4
    num_layers: int = 2
    cg_steps: int | None = None
    mesa_retention_init: float | None = None
    gdn_retention_init: float | None = None
    mesa_lambda: float | None = None
    allow_neg_eigval: bool = False
    mlp_ratio: int = 4


class MockMixer(nn.Module):
    """Pure-torch causal softmax attention — CPU/Mac smoke-test stand-in ONLY.

    Lets the training harness, masking, and metric path be exercised without
    Triton/GPU. It is NOT MesaNet or GatedDeltaNet and must not be used to draw
    any MesaNet-vs-GDN conclusions.

    A causal depthwise short conv precedes attention (mirroring fla's
    ``use_short_conv=True``); it lets adjacent x/y tokens merge so the attention
    can actually pair each x with its label — without it, plain attention has no
    positional cue to do in-context regression.

    forward: hidden_states ``(B, L, H)`` -> ``(B, L, H)`` (causal).
    """

    def __init__(self, hidden_size: int, num_heads: int, conv_size: int = 4) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.conv_size = conv_size
        self.conv = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=conv_size,
            groups=hidden_size, padding=conv_size - 1,  # causal: crop the right
        )
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, **_: object) -> torch.Tensor:
        b, t, _ = hidden_states.shape
        # causal depthwise short conv over time
        z = self.conv(hidden_states.transpose(1, 2))[..., :t].transpose(1, 2)
        q, k, v = self.qkv(z).chunk(3, dim=-1)
        shp = (b, t, self.num_heads, self.head_dim)
        q, k, v = (z.view(shp).transpose(1, 2) for z in (q, k, v))  # (B,h,L,d)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        causal = torch.triu(
            torch.ones(t, t, dtype=torch.bool, device=att.device), diagonal=1
        )
        att = att.masked_fill(causal, float("-inf")).softmax(dim=-1)
        out = (att @ v).transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(out)


class _MixerAdapter(nn.Module):
    """Uniform call interface: ``mixer(x) -> tensor`` regardless of fla's tuple."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor, past_key_values=None, use_cache: bool = False) -> torch.Tensor:
        # past_key_values/use_cache are only exercised by the recurrent decode path
        # (see compare.profile); training/prefill leaves them at the None/False default,
        # which is identical to not passing them at all.
        out = self.inner(hidden_states=x, past_key_values=past_key_values, use_cache=use_cache)
        return out[0] if isinstance(out, tuple) else out


def build_mixing_layer(
    layer: str,
    hidden_size: int,
    num_heads: int,
    layer_idx: int,
    cg_steps: int | None = None,
    mesa_retention_init: float | None = None,
    gdn_retention_init: float | None = None,
    mesa_lambda: float | None = None,
    allow_neg_eigval: bool = False,
) -> nn.Module:
    """Construct the swappable mixing layer, wrapped for a uniform interface.

    Args:
        layer: "mesa" | "gated_deltanet" | "delta_net" | "mock".
        hidden_size: model width.
        num_heads: number of heads; head_dim = hidden_size // num_heads.
        layer_idx: block index (fla layers use it for the kv-cache).
        cg_steps: Mesa CG iterations (the sweep dial); ignored otherwise.
        mesa_retention_init: Mesa decay-gate bias init (see below); Mesa-only.
        gdn_retention_init: GDN target initial per-step decay in (0, 1); GDN-only.
        mesa_lambda: if set, pin MesaNet's ridge regularizer Λ to this value and
            freeze it (stronger Λ -> more regularized, GLA-ward solve); Mesa-only.
        allow_neg_eigval: widen eigenvalues to γ ∈ (−1,1) for state-tracking; GDN-only
            (fla ``GatedDeltaNet(allow_neg_eigval=...)``). Raises for "mesa": fla's
            MesaNet has no negative-eigenvalue path (see below).

    Returns:
        An ``nn.Module`` with ``forward(x: (B,L,H)) -> (B,L,H)``.
    """
    if hidden_size % num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads")
    head_dim = hidden_size // num_heads

    if layer == "mesa":
        from fla.layers.mesa_net import MesaNet  # lazy: needs Triton

        cg = 30 if cg_steps is None else int(cg_steps)
        inner = MesaNet(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            mode="chunk",
            use_output_gate=True,   # per-head output gate (match GDN's use_gate)
            use_short_conv=True,
            layer_idx=layer_idx,
            max_cg_step_training=cg,   # <-- CG-step sweep dial
            max_cg_step_decoding=cg,   # <-- keep decoding consistent with training
        )
        # fla's Mesa decay gate is g = logsigmoid(a_proj(x)); a_proj.bias starts ~0,
        # so the initial per-step state decay is exp(logsigmoid(0)) ~ 0.5 — Mesa
        # forgets half its state per token at init. That is a poor prior for
        # in-context regression (where the whole history should be retained), and
        # at small training budgets Mesa spends it all just learning not to forget.
        # GDN's stock init already retains (Mamba-style tiny dt -> decay ~1), so the
        # comparison is unfair until we equalise. Seeding the bias positive makes
        # Mesa's initial decay ~ exp(logsigmoid(value)) ~ 1; it stays fully trainable.
        if mesa_retention_init is not None:
            with torch.no_grad():
                inner.a_proj.bias.fill_(float(mesa_retention_init))
        # Pin the ridge regularizer Λ = softplus(lambda_params) + lambda_lower_bound
        # to a fixed value and freeze it, so it can be *swept* as the regularization
        # strength. fla's constructor inits lambda_params from lambda_lower_bound and
        # NaNs for lower_bound >= 1, so we set the bound after construction and push
        # lambda_params -> -inf-ward (softplus ~ 0) so Λ ~ mesa_lambda exactly.
        if mesa_lambda is not None:
            with torch.no_grad():
                inner.lambda_lower_bound = float(mesa_lambda)
                inner.lambda_params.fill_(-10.0)   # softplus(-10) ~ 5e-5 ~ 0
            inner.lambda_params.requires_grad_(False)
        # γ ∈ (−1,1) is STRUCTURALLY UNAVAILABLE for MesaNet in fla. The decay gate is
        # g = logsigmoid(a_proj(x)) -> per-step decay exp(g) ∈ (0,1), fed to a log-cumsum
        # chunk kernel (decay reconstructed as exp(cumsum(g))), which cannot represent a
        # negative multiplier; and the read-out solves (H_t+Λ)⁻¹q, which requires
        # (H_t+Λ) positive-definite — exactly the MesaNet-paper App. I restriction. So
        # there is no flag/tanh-swap that makes fla Mesa track state. Raising here makes
        # that the explicit (negative) result rather than a silent (0,1) mislabel.
        if allow_neg_eigval:
            raise NotImplementedError(
                "allow_neg_eigval is not supported for MesaNet: fla's Mesa decay gate is "
                "sigmoid-bounded (γ∈(0,1)) and its solve needs (H+Λ) positive-definite, so "
                "negative eigenvalues are inexpressible (MesaNet paper App. I). This is the "
                "structural asymmetry the parity experiment demonstrates — see notebook 05."
            )
    elif layer == "gated_deltanet":
        from fla.layers.gated_deltanet import GatedDeltaNet  # lazy: needs Triton

        if cg_steps is not None:
            import warnings

            warnings.warn("cg_steps is ignored for gated_deltanet", stacklevel=2)
        inner = GatedDeltaNet(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_v_heads=num_heads,
            head_dim=head_dim,
            expand_v=1.0,           # match Mesa's value_dim == key_dim
            mode="chunk",
            use_gate=True,          # per-head output gate (match Mesa)
            use_short_conv=True,
            layer_idx=layer_idx,
            # γ ∈ (−1,1): fla scales β by 2 so the delta-rule transition (I − β·kkᵀ) can
            # have eigenvalue −1 — the bit-flip parity needs (Grazzi/Sarrof). Verified:
            # GatedDeltaNet.__init__ `allow_neg_eigval` (fla/layers/gated_deltanet.py:97).
            allow_neg_eigval=allow_neg_eigval,
        )
        # GDN's per-step state decay is exp(-exp(A_log) * softplus(a_proj(x) + dt_bias))
        # (verified: fla/ops/gated_delta_rule/gate.py L29,45; chunk.py L411). fla's stock
        # init draws A ~ U(0,16) and dt ~ logU(1e-3,1e-1), so the per-head initial decay
        # is heterogeneous (~0.2-1.0) — NOT uniformly retentive. To match Mesa's near-1
        # init we pin a single target decay `delta`: A_log=0 (so exp(A_log)=1) and
        # dt_bias = softplus^{-1}(-ln delta), giving initial decay = delta at a_proj~0.
        if gdn_retention_init is not None:
            delta = float(gdn_retention_init)
            if not 0.0 < delta < 1.0:
                raise ValueError("gdn_retention_init must be in (0, 1)")
            dt = -math.log(delta)                       # target softplus(dt_bias)
            with torch.no_grad():
                inner.A_log.zero_()
                inner.dt_bias.fill_(math.log(math.expm1(dt)))   # inverse softplus
    elif layer == "delta_net":
        from fla.layers.delta_net import DeltaNet  # lazy: needs Triton

        if cg_steps is not None:
            import warnings

            warnings.warn("cg_steps is ignored for delta_net", stacklevel=2)
        # Ungated DeltaNet: the delta rule WITHOUT the forget/decay gate (the "Gated"
        # in Gated DeltaNet). Matched to GDN in every other respect so the gate is the
        # only difference: expand_k=expand_v=1 -> key_dim=value_dim=hidden_size and
        # head_dim = hidden_size // num_heads (same as Mesa/GDN); per-head output gate
        # and short conv kept on. mesa_/gdn_retention_init do not apply (no decay gate).
        inner = DeltaNet(
            mode="chunk",
            hidden_size=hidden_size,
            expand_k=1.0,
            expand_v=1.0,
            num_heads=num_heads,
            use_beta=True,
            use_gate=True,          # per-head output gate (match Mesa/GDN)
            use_short_conv=True,
            layer_idx=layer_idx,
            allow_neg_eigval=allow_neg_eigval,
        )
    elif layer == "mock":
        inner = MockMixer(hidden_size, num_heads)
    else:
        raise ValueError(
            f"unknown layer {layer!r}; expected 'mesa', 'gated_deltanet', "
            "'delta_net', or 'mock'"
        )

    return _MixerAdapter(inner)


class MLP(nn.Module):
    """Standard 2-layer GELU MLP; shared by both conditions."""

    def __init__(self, hidden_size: int, mlp_ratio: int) -> None:
        super().__init__()
        inner = hidden_size * mlp_ratio
        self.fc1 = nn.Linear(hidden_size, inner)
        self.fc2 = nn.Linear(inner, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm residual block: mix then MLP. Identical across conditions."""

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.hidden_size)
        self.mixer = build_mixing_layer(
            cfg.layer, cfg.hidden_size, cfg.num_heads, layer_idx, cfg.cg_steps,
            cfg.mesa_retention_init, cfg.gdn_retention_init, cfg.mesa_lambda,
            cfg.allow_neg_eigval,
        )
        self.norm2 = nn.RMSNorm(cfg.hidden_size)
        self.mlp = MLP(cfg.hidden_size, cfg.mlp_ratio)

    def forward(self, x: torch.Tensor, past_key_values=None, use_cache: bool = False) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x), past_key_values=past_key_values, use_cache=use_cache)
        x = x + self.mlp(self.norm2(x))
        return x


class SequenceModel(nn.Module):
    """Tiny sequence model: in-proj -> N blocks -> norm -> task head.

    forward input/output by config:
      continuous + regression     : x ``(B,L,input_dim)`` -> ``(B,L)`` scalars.
      tokens     + classification : x ``(B,L)`` long       -> ``(B,L,num_classes)``.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        if cfg.input_kind == "continuous":
            if cfg.input_dim is None:
                raise ValueError("input_dim required for continuous inputs")
            self.in_proj: nn.Module = nn.Linear(cfg.input_dim, cfg.hidden_size)
        elif cfg.input_kind == "tokens":
            if cfg.vocab is None:
                raise ValueError("vocab required for token inputs")
            self.in_proj = nn.Embedding(cfg.vocab, cfg.hidden_size)
        else:
            raise ValueError(f"unknown input_kind {cfg.input_kind!r}")

        self.blocks = nn.ModuleList(
            [Block(cfg, layer_idx=i) for i in range(cfg.num_layers)]
        )
        self.final_norm = nn.RMSNorm(cfg.hidden_size)

        if cfg.output_kind == "regression":
            self.head: nn.Module = nn.Linear(cfg.hidden_size, 1)
        elif cfg.output_kind == "classification":
            if cfg.num_classes is None:
                raise ValueError("num_classes required for classification")
            self.head = nn.Linear(cfg.hidden_size, cfg.num_classes)
        else:
            raise ValueError(f"unknown output_kind {cfg.output_kind!r}")

    def forward(self, x: torch.Tensor, past_key_values=None, use_cache: bool = False) -> torch.Tensor:
        # past_key_values/use_cache thread an fla recurrent Cache through the blocks for
        # token-by-token decoding (compare.profile); default None/False = plain forward.
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, past_key_values=past_key_values, use_cache=use_cache)
        h = self.final_norm(h)
        out = self.head(h)
        if self.cfg.output_kind == "regression":
            return out.squeeze(-1)  # (B, L)
        return out  # (B, L, num_classes)


def count_params(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # CPU smoke test of the shared backbone via the mock mixer (no fla/Triton).
    cfg = ModelConfig(
        layer="mock", input_kind="continuous", output_kind="regression",
        input_dim=5, hidden_size=64, num_heads=4, num_layers=2,
    )
    model = SequenceModel(cfg)
    x = torch.randn(3, 12, 5)
    y = model(x)
    print("=== model.py smoke test (mock mixer) ===")
    print(f"continuous/regression: in {tuple(x.shape)} -> out {tuple(y.shape)}")
    print(f"params: {count_params(model):,}")

    cfg2 = ModelConfig(
        layer="mock", input_kind="tokens", output_kind="classification",
        vocab=16, num_classes=16, hidden_size=64, num_heads=4, num_layers=2,
    )
    m2 = SequenceModel(cfg2)
    xt = torch.randint(0, 16, (3, 10))
    y2 = m2(xt)
    print(f"tokens/classification: in {tuple(xt.shape)} -> out {tuple(y2.shape)}")
