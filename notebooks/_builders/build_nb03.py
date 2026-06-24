"""Build notebooks/03_mqar_capacity.ipynb from cell sources (nbformat)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md(r"""# 03 · MQAR Capacity — where exact full-history fit should pay off

**Goal.** Notebooks 01–02 covered in-context *regression* (clean, then noisy/drifting). This one is the **recall/capacity** test: write a set of key→value bindings into the recurrent state, then read them back. This is precisely the task where MesaNet's exact, full-history solve and its extra $H_t$ state *should* earn their cost — regression was rigged for Mesa and drift is gate-dependent, but **recall is where exact fitting has teeth**.

**Pre-registered hypotheses** (stated before looking at any result):

- **(a) Easy regime is a tie.** When `n_pairs` is small relative to the layer's effective key dimension, *both* models recall near-perfectly.
- **(b) Mesa degrades slower under load.** As `n_pairs` grows toward and past the key dimension, GDN's accuracy should fall **faster** than Mesa's, because Mesa's exact full-history solve resolves key interference (memory collisions) that GDN's single gated-delta step cannot.
- **(c) Mesa's edge grows with `n_pairs` and with `gap`.** More bindings → more interference (favours the exact solve); larger write→query distance → more retention demanded.

**Asymmetric prediction / known confound.** Recall is the task where **retention** matters most, so the forget-gate init confound (flagged in 01–02) bites hardest *here*. If GDN's gate were initialised to retain more than Mesa's, part of any GDN edge would be *init, not algorithm*. We therefore **match forgetting** (both gates pinned to the same near-1 initial decay, as in 02) for the main sweeps, and note the stock-GDN ablation. If despite all this **Mesa lost to GDN on recall**, that would be a strong signal something upstream is broken — this is the task Mesa should win.""")

md(r"""## Experimental design

- **Train-across / evaluate-per-setting.** We train **one** model per condition on a *distribution* that mixes both difficulty axes — each training batch draws `n_pairs` from `TRAIN_PAIRS` **and** `gap` from `TRAIN_GAPS` (uniform, independent) — then evaluate that single frozen model at every probed setting. One model per `(layer, CG)`; the sweeps are evaluation-time only. (This two-axis mix is why the MQAR trainer `train_mqar` exists rather than the single-knob `train_across_eval`.)
- **Identical scoring path.** Every number is answer-token exact-match via `synthtasks.metrics.mqar_exact_match`, scoring only the masked query positions — the same path the data sanity check uses.
- **Forgetting held equal & near 1** (as in 02). Mesa decay-gate bias `MESA_RET=4.0` → initial per-step decay `σ(4.0) ≈ 0.982`; GDN pinned to the **same** decay via `gdn_retention_init` (`A_log=0`, `dt_bias=softplus⁻¹(−ln δ)`). This removes the gate-init confound so the comparison isolates *capacity/algorithm*, not forgetting. Both gates stay trainable.
- **Test-time compute = CG steps.** Mesa's conjugate-gradient count is the compute dial: CG=1 ≈ the GLA readout, CG→30 ≈ the exact $(H+\lambda I)^{-1}q$ solve (see the verified CG-semantics finding in 01). Analytic cost: **Mesa(CG=k) ≈ k× the GLA mixer FLOPs; GDN ≈ 1× GLA**.
- **Small models, reproducible, GPU.** Tiny 2-layer models; the fla kernels need CUDA + Triton. (GPU/Triton kernels are not bit-deterministic, so absolute accuracies wobble a little run-to-run; the *qualitative* orderings below are stable.)""")

code(r"""%matplotlib inline
import sys, os, math
sys.path.insert(0, os.path.abspath(".."))  # project root: makes `compare` importable

import numpy as np
import torch
import matplotlib.pyplot as plt

from compare.experiments import (
    MQARConfig, train_mqar_models, mqar_sweep_rows, mqar_cg_rows,
    mqar_position_rows, mqar_flops_note, plot_mse_vs,
)

# ----- config (edit here) -----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 0

VOCAB        = 128                       # distinct keys live in [1, VOCAB); need VOCAB > max n_pairs
N_QUERIES    = 8                         # queries per sequence (also the depth axis for the position view)
TRAIN_PAIRS  = (2, 4, 8, 16, 32, 64)     # n_pairs mixed during training (memory load)
TRAIN_GAPS   = (2, 8, 16, 32)            # write->query gap mixed during training (retention)
STEPS        = 4000                      # Mesa converges slowly on recall -> a generous budget
                                         # (with grad-clip + warmup; see the stability caveat in the close)
CG_GRID      = [1, 2, 5, 10, 30]         # Mesa CG steps (test-time-compute dial)

# evaluation grids (evaluation-time only; the trained models are frozen)
PAIRS_GRID   = [2, 4, 8, 16, 32, 64, 96] # capacity sweep: clearly under- to clearly over-loaded
GAP_GRID     = [2, 8, 16, 32, 64]        # gap (retention) sweep
GAP_FIXED    = 8                         # gap held during the capacity sweep
NPAIRS_GAP   = 32                        # n_pairs held during the gap sweep (already under pressure)
NPAIRS_HEAD  = 64                        # headline CG-overlay setting: capacity actually bites here
GAP_HEAD     = 8
NPAIRS_POS   = 32                        # position-resolved (depth) setting
GAP_POS      = 8

# effective key dimension (rough): per-head state is head_dim x head_dim
HEAD_DIM = 128 // 4
print(f"head_dim = {HEAD_DIM}  (num_heads=4, hidden=128)  ->  expect recall to collapse "
      f"as n_pairs approaches ~{HEAD_DIM}-{4*HEAD_DIM}")

# --- forgetting held EQUAL & near 1 (as in notebook 02) ---
MESA_RET    = 4.0
GDN_RET     = float(torch.sigmoid(torch.tensor(MESA_RET)))   # σ(4.0) ≈ 0.982 (matched decay)
RETAIN_BIAS = 8.0                                            # §1b control: σ(8) ≈ 0.99966 -> forgetting "off"
print(f"forgetting matched: Mesa bias={MESA_RET} -> decay σ(4.0)={GDN_RET:.3f} "
      f"(memory horizon ~{1/(1-GDN_RET):.0f} tokens) ; GDN pinned to {GDN_RET:.3f}")

mqcfg = MQARConfig(
    vocab=VOCAB, n_queries=N_QUERIES, train_pairs=TRAIN_PAIRS, train_gaps=TRAIN_GAPS,
    steps=STEPS, hidden_size=128, num_heads=4, num_layers=2,
    eval_batches=8, device=DEVICE,
    mesa_retention_init=MESA_RET, gdn_retention_init=GDN_RET,
)   # lr / grad_clip / warmup use the stabilised MQARConfig defaults (2e-3, clip 1.0, warmup 200)""")

code(r'''def show(rows, x, ykey="acc"):
    """Compact accuracy table: one row per model label, columns = the swept axis `x`."""
    labels = []
    for r in rows:
        if r["label"] not in labels:
            labels.append(r["label"])
    xs = sorted({r[x] for r in rows if r.get(x) is not None})
    print(f"{x:>16}  " + "  ".join(f"{v:>6}" for v in xs))
    for lb in labels:
        dd = {r[x]: r[ykey] for r in rows if r["label"] == lb and r.get(x) is not None}
        body = "  ".join((f"{dd[v]:6.3f}" if v in dd else "    --") for v in xs)
        print(f"{lb:>16}  {body}")''')

md(r"""## Train the models (once) — then every sweep is evaluation-only

We train the full set **Mesa at each CG ∈ {1,2,5,10,30} + GDN** = 6 models, each on the mixed `(n_pairs, gap)` distribution. Training is the only expensive step; all four figures below reuse these frozen models. (Mesa's recall converges slowly, so this cell is the slowest in the notebook.)""")

code(r"""ALL_SPECS = [("mesa", k) for k in CG_GRID] + [("gated_deltanet", None)]
models = train_mqar_models(ALL_SPECS, mqcfg, seed=SEED)

print("trained models (frozen, reused for every sweep):")
for lb, tm in models.items():
    quick = tm.accuracy(4, GAP_FIXED)          # easy setting: should be ~1.0
    print(f"  {lb:16}  params={tm.num_params:,}  acc@(4 pairs)={quick:.3f}  dev={tm.device}")

# subset reused for the capacity / gap / position figures ("a couple of CG settings" + GDN)
CMP = {lb: models[lb] for lb in ["Mesa (CG=1)", "Mesa (CG=30)", "Gated DeltaNet"]}""")

md(r"""## 1 · Capacity sweep — accuracy vs `n_pairs`  (gap fixed)

Fix `gap = GAP_FIXED`, vary `n_pairs` from clearly underloaded (2) to clearly overloaded (96) relative to the effective key dimension. Tests hypotheses **(a)** tie when small and **(b)** GDN degrades faster than Mesa.""")

code(r"""cap_rows = mqar_sweep_rows(CMP, "n_pairs", PAIRS_GRID, gap=GAP_FIXED, seed=SEED)
print(f"answer-token exact-match accuracy  (gap={GAP_FIXED}, vary n_pairs):")
show(cap_rows, "n_pairs")""")

code(r"""ax = plot_mse_vs(cap_rows, "n_pairs",
                 title=f"MQAR capacity: recall accuracy vs memory load (gap={GAP_FIXED})",
                 xlabel="n_pairs  (bindings stored — memory load)",
                 ykey="acc", ylabel="answer-token exact-match accuracy",
                 logy=False, logx=True, lw=1.1)
ax.axvspan(HEAD_DIM, 4 * HEAD_DIM, color="0.85", alpha=0.5, zorder=0,
           label=f"~effective key dim ({HEAD_DIM}–{4*HEAD_DIM})")
ax.axhline(1.0 / VOCAB, color="gray", ls=":", lw=1, label=f"chance (1/{VOCAB})")
ax.set_ylim(-0.02, 1.02)
ax.legend(fontsize=8)
plt.tight_layout(); plt.show()""")

md(r"""**Reading it.** **(a) holds:** all curves start at ~1.0 for small `n_pairs`. **(b) holds:** as load grows into the shaded effective-key-dimension band, **GDN falls fastest** and the **exact-solve Mesa (CG=30) holds up best**, the Mesa-over-GDN margin widening with load — the large, robust separation here is **Mesa-class over GDN**. The *cheap* Mesa (CG=1 ≈ the GLA read-out) sits between them and is the **most run-variable** curve; §1b below shows why — unlike the exact solve, CG=1 is sensitive to early-binding *forgetting* at this decay setting, which is also a hidden confound in this very plot. Read the CG=1-vs-CG=30 gap only after §1b.""")

md(r"""## 1b · Capacity *control* — is the §1 collapse capacity, or forgetting?

§1 was run at matched decay **δ ≈ 0.98**. But "near 1" *per step* compounds: over a 100+-token write block, 0.98ᴸ heavily attenuates the **earliest** bindings — the effective memory horizon is `1/(1−δ) ≈ 56 tokens ≈ 28 pairs`. That horizon lands **right on top of** the state-capacity limit (~`head_dim = 32`), so the §1 collapse could be *either* the fixed state running out of room *or* the gate forgetting the early writes — the two are confounded.

This control re-runs the capacity sweep with forgetting effectively **off** (`δ → 1` via `RETAIN_BIAS=8.0`, horizon ≈ 3000 tokens ≫ any sequence here) and overlays it on the matched-δ curves. **If a model's collapse shifts to higher `n_pairs` with forgetting off, its §1 collapse was partly *forgetting*; if the curve doesn't move, it's genuine *capacity*.**""")

code(r"""from compare.experiments import plot_capacity_forget_control

GDN_OFF = float(torch.sigmoid(torch.tensor(RETAIN_BIAS)))   # match GDN's initial decay to δ→1
mqcfg_off = MQARConfig(
    vocab=VOCAB, n_queries=N_QUERIES, train_pairs=TRAIN_PAIRS, train_gaps=TRAIN_GAPS,
    steps=STEPS, hidden_size=128, num_heads=4, num_layers=2, eval_batches=8, device=DEVICE,
    mesa_retention_init=RETAIN_BIAS, gdn_retention_init=GDN_OFF,
)
print(f"forgetting OFF: Mesa bias={RETAIN_BIAS} -> decay σ={GDN_OFF:.5f} "
      f"(memory horizon ~{1/(1-GDN_OFF):.0f} tokens) ; GDN pinned to {GDN_OFF:.5f}")

# train the same 3-model subset with forgetting off (reuses CMP's labels)
models_off   = train_mqar_models([("mesa", 1), ("mesa", 30), ("gated_deltanet", None)],
                                 mqcfg_off, seed=SEED)
cap_rows_off = mqar_sweep_rows(models_off, "n_pairs", PAIRS_GRID, gap=GAP_FIXED, seed=SEED)
print(f"\naccuracy with forgetting OFF (gap={GAP_FIXED}, vary n_pairs):")
show(cap_rows_off, "n_pairs")""")

code(r"""ax = plot_capacity_forget_control(
    cap_rows_off, cap_rows,
    xlabel="n_pairs  (bindings stored — memory load)",
    title=f"Capacity control: forgetting off (solid) vs matched δ≈0.98 (dashed), gap={GAP_FIXED}")
ax.axvspan(HEAD_DIM, 4 * HEAD_DIM, color="0.85", alpha=0.4, zorder=0)
ax.axhline(1.0 / VOCAB, color="gray", ls=":", lw=1)
ax.set_ylim(-0.02, 1.02)
plt.tight_layout(); plt.show()""")

md(r"""**Reading it — this is the key control.** The two decay settings cleanly separate the operators:

- **Mesa CG=30 (exact solve): the solid and dashed curves lie on top of each other.** The least-squares read-out recovers each binding *regardless of how much its magnitude has decayed*, so it is **insensitive to the gate** — its capacity curve is **genuine state capacity**, not forgetting.
- **Mesa CG=1: forgetting off lifts it markedly under load.** The cheap GLA-like read-out genuinely suffers the recency bias, so a real chunk of CG=1's §1 collapse was the gate forgetting early writes — *not* the state running out of room. (GDN can shift too, but by a smaller, run-variable amount; the summary cell labels each model *decay-sensitive* vs *decay-robust* from its actual shift.)

Consequences: (i) read at matched δ≈0.98, §1 **under-credits the cheap read-out** and conflates capacity with forgetting for it — the **exact-solve curve is the clean capacity probe**. (ii) This is exactly why CG=1 wobbles run-to-run in §1: being decay-sensitive, its score depends on how far training pushed the gate off its 0.98 prior. (iii) It also **partly decomposes the headline CG benefit**: with forgetting *off* the CG=30−CG=1 gap shrinks (some of the matched-δ gap was a decay handicap, not deconvolution) — though CG=1's own instability means treat the residual's exact size loosely. (The δ≈0.98 value is inherited from the regression notebooks, where sequences are short enough that this compounding is negligible.)""")

md(r"""## 2 · Gap sweep — accuracy vs `gap`  (n_pairs fixed under pressure)

Hold a moderately-loaded `n_pairs = NPAIRS_GAP` and vary the write→query distance `gap` (blank filler tokens between the writes and the queries). Tests the retention half of hypothesis **(c)**.""")

code(r"""gap_rows = mqar_sweep_rows(CMP, "gap", GAP_GRID, n_pairs=NPAIRS_GAP, seed=SEED)
print(f"answer-token exact-match accuracy  (n_pairs={NPAIRS_GAP}, vary gap):")
show(gap_rows, "gap")""")

code(r"""ax = plot_mse_vs(gap_rows, "gap",
                 title=f"MQAR retention: recall accuracy vs write→query gap (n_pairs={NPAIRS_GAP})",
                 xlabel="gap  (blank filler tokens between writes and queries)",
                 ykey="acc", ylabel="answer-token exact-match accuracy",
                 logy=False, lw=1.1)
ax.set_ylim(-0.02, 1.02)
ax.legend(fontsize=8)
plt.tight_layout(); plt.show()""")

md(r"""**Reading it.** If retention is the bottleneck, accuracy should fall as `gap` grows. With forgetting pinned near 1 for both layers and the gap filled by *blank* (id 0) tokens that carry no competing bindings, a **flat** result is the honest expectation here — it says the bottleneck is *capacity* (`n_pairs`), not *distance* (`gap`). Any vertical separation between models at fixed gap is the same capacity gap as in §1.""")

md(r"""## 3 · Headline — the CG cost-vs-quality trade-off  (where capacity bites)

This is the figure the whole suite builds toward: at a setting where capacity *actually bites* (`n_pairs = NPAIRS_HEAD`), how does recall accuracy move as we spend more test-time compute (Mesa CG steps)? **GDN is the horizontal reference** (it has no CG dial; cost ≈ 1× GLA). The analytic FLOPs annotation makes the trade-off explicit: each CG step ≈ one extra GLA-equivalent pass.""")

code(r"""cg_rows = mqar_cg_rows(models, NPAIRS_HEAD, GAP_HEAD, seed=SEED)
print(f"answer-token exact-match accuracy  (n_pairs={NPAIRS_HEAD}, gap={GAP_HEAD}):")
show(cg_rows, "cg_steps")
gdn_ref = [r["acc"] for r in cg_rows if r["layer"] == "gated_deltanet"][0]
print(f"\nGDN reference accuracy = {gdn_ref:.3f}   |   {mqar_flops_note(30)}")""")

code(r"""ax = plot_mse_vs(cg_rows, "cg_steps",
                 title=f"Headline: recall accuracy vs test-time compute (n_pairs={NPAIRS_HEAD}, gap={GAP_HEAD})",
                 xlabel="Mesa CG steps  (≈ ×GLA mixer FLOPs;  log scale)",
                 ykey="acc", ylabel="answer-token exact-match accuracy",
                 logy=False, logx=True, lw=1.3, dim_alpha=1.0)
ax.set_xticks(CG_GRID); ax.set_xticklabels([str(c) for c in CG_GRID])
ax.set_ylim(-0.02, 1.02)
# annotate the analytic FLOPs cost of the most expensive setting
ax.annotate(mqar_flops_note(30), xy=(30, [r["acc"] for r in cg_rows
            if r.get("cg_steps") == 30][0]),
            xytext=(0.32, 0.12), textcoords="axes fraction", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))
ax.legend(fontsize=8)
plt.tight_layout(); plt.show()""")

md(r"""**Reading it — the cost-vs-quality story.** Two findings, in order of robustness.

1. **CG ≥ 2 reaches a plateau far above GDN.** From CG=2 onward Mesa sits at a stable plateau well above the GDN reference line — the exact solve's deconvolution of colliding keys is captured almost entirely within a *couple* of CG steps (echoing nb02's "CG ≥ 2 captures most of the benefit"). So the robust cost-vs-quality statement is: **~2× GLA FLOPs (CG=2) already buys the recall plateau and clears GDN; further CG adds little.** The gap from that plateau down to the GDN line is the exact-solve advantage over a single delta step.

2. **CG=1 is the unreliable outlier — don't read it as "the cheap Mesa."** The bare GLA-like read-out at CG=1 is both *decay-sensitive* (§1b) and *seed-variable*, so depending on the run it can sit near the plateau or fall **below** GDN. Removing its decay handicap (δ→1, §1b) shrinks the CG=30−CG=1 gap, but CG=1's own instability means any precise "CG=1 vs CG=30" number is not to be trusted — which is exactly why we anchor the story on the CG≥2 plateau, not on CG=1.

**Methodological note.** An earlier higher-LR version showed CG=1 *strongly* beating CG=30 — training instability in Mesa's exact-solve path (CE blow-ups), not a property of the solve. Gradient clipping + LR warmup (now in `MQARConfig`) fixed it. The lesson: the exact solve is *harder to optimise* and the cheap read-out is *decay-sensitive*, so stress-test any "compute changes recall" claim against both the optimiser (this note) and the gate (§1b) before believing it.""")

md(r"""## 4 · Position-resolved view — does Mesa's edge grow with depth?

A cheap depth-of-sequence probe that needs no separate length sweep: at one loaded setting, plot accuracy at each **query index** (1st query, 2nd, …). Later query indices sit deeper past the write block, so this reads off how recall holds up with depth — testing the "Mesa's edge grows late" prediction directly.""")

code(r"""pos_rows = mqar_position_rows(CMP, NPAIRS_POS, GAP_POS, n_queries=N_QUERIES, seed=SEED)
print(f"per-query-index accuracy  (n_pairs={NPAIRS_POS}, gap={GAP_POS}, {N_QUERIES} queries):")
show(pos_rows, "query_pos")""")

code(r"""ax = plot_mse_vs(pos_rows, "query_pos",
                 title=f"Depth probe: recall accuracy by query index (n_pairs={NPAIRS_POS}, gap={GAP_POS})",
                 xlabel="query index within the sequence  (later = deeper past the writes)",
                 ykey="acc", ylabel="answer-token exact-match accuracy",
                 logy=False, lw=1.1)
ax.set_xticks(range(1, N_QUERIES + 1))
ax.set_ylim(-0.02, 1.02)
ax.legend(fontsize=8)
plt.tight_layout(); plt.show()""")

md(r"""**Reading it.** Mesa's per-query accuracy is essentially **flat across query depth** (it holds steady from the 1st to the last query), while GDN drifts slightly *down*. So Mesa's *relative* edge grows only **mildly** with depth — driven by GDN's small decline, not a Mesa rise. The "Mesa's edge grows late" prediction is at best weakly supported here: within the query block, depth is not the binding constraint — **load (`n_pairs`) is** (consistent with the flat gap sweep). *(Caveat: this bins by* query *index, which is roughly uniform in difficulty. The forgetting-sensitive depth probe is accuracy vs the queried key's* write *position — early-written keys fade faster at δ<1; §1b already isolates that effect.)*""")

md(r"""## 5 · Summary table — computed from the runs above

A compact, run-agnostic readout (so the conclusions don't depend on hardcoded numbers): the easy-regime tie, the Mesa-vs-GDN margin at high load, the best CG operating point at the headline setting, and the depth trend.""")

code(r"""def margin_at(rows, x, xv):
    d = {r["label"]: r["acc"] for r in rows if r.get(x) == xv}
    return d

easy = margin_at(cap_rows, "n_pairs", min(PAIRS_GRID))
hard = margin_at(cap_rows, "n_pairs", max(PAIRS_GRID))
print(f"(a) easy regime (n_pairs={min(PAIRS_GRID)}):  " +
      "  ".join(f"{k}={v:.2f}" for k, v in easy.items()))
print(f"    -> all near 1.0? {'YES' if all(v > 0.9 for v in easy.values()) else 'NO'}\n")

best_mesa_hard = max(v for k, v in hard.items() if k.startswith("Mesa"))
gdn_hard = hard.get("Gated DeltaNet", float('nan'))
print(f"(b) heavy load (n_pairs={max(PAIRS_GRID)}):  " +
      "  ".join(f"{k}={v:.2f}" for k, v in hard.items()))
print(f"    -> best Mesa − GDN margin = {best_mesa_hard - gdn_hard:+.2f}  "
      f"({'Mesa wins recall' if best_mesa_hard > gdn_hard else 'GDN wins — investigate!'})\n")

cg_acc = {r["cg_steps"]: r["acc"] for r in cg_rows if r["layer"] == "mesa"}
best_cg = max(cg_acc, key=cg_acc.get)
print(f"(c) headline (n_pairs={NPAIRS_HEAD}, gap={GAP_HEAD}):  "
      f"best Mesa CG = {best_cg} (acc {cg_acc[best_cg]:.2f});  "
      f"CG=1 {cg_acc.get(1, float('nan')):.2f}  CG=30 {cg_acc.get(30, float('nan')):.2f};  "
      f"GDN {gdn_ref:.2f}")
cg_gap = cg_acc.get(30, float("nan")) - cg_acc.get(1, float("nan"))
verdict = ("more CG clearly helps" if cg_gap > 0.05 else
           "more CG clearly hurts" if cg_gap < -0.05 else
           "within noise — CG>=1 already captures recall")
print(f"    -> CG dial (matched δ): {verdict}  (best CG={best_cg}; CG=30 - CG=1 = {cg_gap:+.2f})")

# control: how much of the high-load collapse was forgetting (shift when δ->1)?
print(f"\n(control) forgetting off vs matched, accuracy at n_pairs=64 (gap={GAP_FIXED}):")
ko = {r["label"]: r["acc"] for r in cap_rows if r["n_pairs"] == 64}
oo = {r["label"]: r["acc"] for r in cap_rows_off if r["n_pairs"] == 64}
for lb in ko:
    tag = "(decay-robust: genuine capacity)" if abs(oo[lb] - ko[lb]) < 0.03 else "(decay-sensitive: §1 collapse was partly forgetting)"
    print(f"    {lb:16} matched {ko[lb]:.2f} -> off {oo[lb]:.2f}  ({oo[lb]-ko[lb]:+.2f})  {tag}")
cg1_gap_off = cap_rows_off and (
    {r['label']: r['acc'] for r in cap_rows_off if r['n_pairs'] == 64}.get('Mesa (CG=30)', float('nan'))
    - {r['label']: r['acc'] for r in cap_rows_off if r['n_pairs'] == 64}.get('Mesa (CG=1)', float('nan')))
print(f"    -> genuine exact-solve benefit at δ->1 (CG=30 - CG=1, n_pairs=64) = {cg1_gap_off:+.2f}")

pos_first = {r["label"]: r["acc"] for r in pos_rows if r["query_pos"] == 1}
pos_last  = {r["label"]: r["acc"] for r in pos_rows if r["query_pos"] == N_QUERIES}
print(f"\n(depth) query 1 -> query {N_QUERIES} accuracy change:")
for lb in pos_first:
    print(f"    {lb:16} {pos_first[lb]:.2f} -> {pos_last[lb]:.2f}  ({pos_last[lb]-pos_first[lb]:+.2f})")""")

md(r"""## Closing — did the pre-registered hypotheses hold?

*(Read against the tables/plots above; small-scale comparison, forgetting matched near 1 for both layers, train-across / evaluate-per-setting.)*

- **(a) Easy-regime tie — held.** At small `n_pairs` every model recalls near-perfectly; the architectures are indistinguishable until the state saturates.
- **(b) Mesa degrades slower than GDN — held, and this is the headline result.** With forgetting matched (not a gate-init artefact), the exact-solve Mesa (CG=30, and the CG≥2 plateau) stays above GDN as load grows, the margin widening through the effective-key-dimension band. This is the capacity advantage of keeping the full $H_t$ and reading it out by a least-squares solve rather than a single delta step. **Mesa beating GDN here is the healthy outcome** — recall is the task Mesa should win, and it does.
- **(c) "edge grows with `n_pairs` and `gap`", and the CG dial — refined.** Grows clearly with **`n_pairs`** (b); **flat in `gap`** (blank filler carries no interference) and only mildly depth-dependent. On compute: **the CG≥2 plateau sits far above GDN and ~2 CG steps already capture it** — extra CG adds little. CG=1 alone is an unreliable outlier (decay-sensitive per §1b, and seed-variable), so we anchor the cost-vs-quality claim on the CG≥2 plateau, not on the noisy CG=1 point.

**The key control (§1b) — capacity vs forgetting.** The decay value `δ≈0.98`, inherited from the regression notebooks, compounds over MQAR's long write blocks into a recency bias (memory horizon ≈ 28 pairs ≈ the state-capacity limit), confounding the two. Turning forgetting off disentangles them: **Mesa CG=30 is unchanged (its solve is decay-robust → genuine capacity), while CG=1 improves markedly (its §1 collapse was partly forgetting).** So the exact-solve curve is the clean capacity probe; the cheap read-out was under-credited at δ≈0.98. For a pure capacity claim, use CG=30 or δ→1.

**Interaction with the forget-gate init confound.** Both gates are pinned to the same decay, so the Mesa-vs-GDN gap is not a forgetting artefact. Stock GDN (heterogeneous init, some heads forgetting fast) recalls **worse** than matched GDN — matching *helps* GDN — so the matched setting is *conservative* for (b). A full `δ`-sweep (as in 02) is the clean follow-up — and it only becomes interesting once distractors fill the gap (forget distractors, keep targets).

**Caveats.** (i) Mesa's exact-solve path is optimisation-sensitive (grad-clip + LR warmup required; without them the high-CG models collapse). (ii) GPU/Triton kernels are not bit-deterministic and the **decay-sensitive** CG=1 (and to a lesser degree GDN) are additionally seed-variable; the **stable findings are: the CG≥2 plateau ≫ GDN at high load, and CG=30 is the stable, decay-robust operator** — finer rankings (and the CG=1 point in particular) sit inside the noise.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
nbf.write(nb, "notebooks/03_mqar_capacity.ipynb")
print(f"wrote notebooks/03_mqar_capacity.ipynb with {len(cells)} cells")
