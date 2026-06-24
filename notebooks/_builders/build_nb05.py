"""Build notebooks/05_delta_cg_interaction.ipynb from cell sources (nbformat)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md(r"""# 05 · δ × CG interaction — do Mesa's two levers compose?

The suite has studied two MesaNet levers **separately**: the **forget gate** δ (retention; 02, 04) and the **solve depth** CG (test-time compute; 03). This notebook maps their **joint surface** on the one task where each lever moves the result strongly — **distractor-MQAR (04)** — and asks whether they are *separable* or *interactive*.

Why this task (not drift): on distractor-MQAR both levers do real work on the *same* sub-problem — keeping the (older) target bindings retrievable through the (newer) distractor interference. δ is monotone-strong here (retention best, 04 §2) and CG has the plateau-vs-collapse structure (04 §1). Drift (02) has δ effects too, but its surface is likely separable; distractor-MQAR is where the levers may genuinely interact.

**Pre-registered hypotheses** (stated before running):

- **A — separable.** The optimal δ is ~constant across CG: solve quality and retention act independently. Confirms the clean "independent levers" framing.
- **B — interactive.** The optimal δ depends on CG — e.g. higher CG *reduces* the need for aggressive retention (the exact solve can disentangle a partially-decayed state), or the two compose so you need *both*. A real architectural finding about how Mesa's mechanisms combine.

**Either outcome is informative** — separable = clean framing; interactive = a finding. The surface should also be **consistent with the 04 one-row hint**: at the fixed setting, CG=30 was *decay-robust* while CG=1 was *decay-sensitive*. This sweep asks whether that single slice generalizes across the whole (δ, CG) plane.""")

md(r"""## Design

- **One trained model per (δ, CG) cell.** Each cell is *structurally different* — both the gate init (`mesa_retention_init = logit(δ)`) and the solve depth (`cg_steps`) differ — so they cannot share weights. 6 δ × 5 CG = **30 Mesa models**, plus **6 GDN reference models** (one per δ, no CG dial) = 36 trainings.
- **Fixed task = the 04 distractor slice.** `n_pairs`, `gap`, `n_distractors` reuse 04's δ-sweep setting (where interference bites and CG=30/CG=1 separated) rather than new values. Training mixes distractors (`train_distractors`) exactly as 04; evaluation is at the fixed `n_distractors`.
- **Matched-init convention.** δ enters Mesa via `mesa_retention_init = logit(δ)` (a_proj.bias) and GDN via `gdn_retention_init = δ` (`A_log=0`, `dt_bias`) — the same parametrization verified in 02–04. CG via `cg_steps` (fla `max_cg_step_{training,decoding}`; CG semantics verified in 03). All in `compare.experiments.mqar_delta_cg_sweep`; the notebook only orchestrates and plots.
- **Cost-aware.** This is the most expensive experiment in the suite — the config cell prints the **total cell count and estimated wall-clock** so the per-cell budget can be chosen *before* the sweep runs.""")

code(r"""%matplotlib inline
import sys, os, math
sys.path.insert(0, os.path.abspath(".."))  # project root: makes `compare` importable

import numpy as np
import torch
import matplotlib.pyplot as plt

from compare.experiments import MQARConfig, mqar_delta_cg_sweep, plot_delta_cg_heatmap, plot_mse_vs

# ----- config (edit here) -----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 0

# fixed distractor-MQAR setting (reused verbatim from 04's δ-sweep slice)
VOCAB, N_QUERIES   = 128, 4
NPAIRS, GAP        = 8, 2
N_DISTRACTORS      = 16                       # interference bites here; CG=30 robust / CG=1 sensitive in 04
TRAIN_DIST         = (0, 8, 16, 32, 64)       # distractor counts mixed during training (as 04)

# the 2-D grid
DELTA_GRID         = [0.5, 0.8, 0.9, 0.95, 0.98, 1.0]   # initial decay δ (1 = retain everything)
CG_GRID            = [1, 2, 5, 10, 30]                  # Mesa solve depth

# per-cell training budget (the lever to tune for total cost)
PER_CELL_STEPS     = 2000                     # 04 used 3000/point; lower = cheaper, shape still readable

mqcfg = MQARConfig(
    vocab=VOCAB, n_queries=N_QUERIES, train_pairs=(NPAIRS,), train_gaps=(GAP,),
    train_distractors=TRAIN_DIST, steps=PER_CELL_STEPS, hidden_size=128, num_heads=4,
    num_layers=2, eval_batches=8, device=DEVICE,
)   # δ is set per cell by the sweep; lr/grad_clip/warmup use the stabilised MQARConfig defaults

# ----- compute estimate (surfaced BEFORE the sweep) -----
N_MESA = len(DELTA_GRID) * len(CG_GRID)
N_GDN  = len(DELTA_GRID)
N_CELLS = N_MESA + N_GDN
SEC_PER_1K = 15.0                             # calibration: 04 measured ~43s/model @ 3000 steps
est_min = N_CELLS * (PER_CELL_STEPS / 1000.0) * SEC_PER_1K / 60.0
print(f"grid: {len(DELTA_GRID)} δ × {len(CG_GRID)} CG = {N_MESA} Mesa + {N_GDN} GDN = {N_CELLS} trained models")
print(f"per-cell steps = {PER_CELL_STEPS}")
print(f"estimated wall-clock ≈ {est_min:.0f} min on this GPU  (~{est_min*60/N_CELLS:.0f}s/cell; CG=30 cells run slower)")
print(f"fixed setting: n_pairs={NPAIRS}, gap={GAP}, n_distractors={N_DISTRACTORS}, vocab={VOCAB}")""")

md(r"""## Run the sweep (the expensive cell)

⚠ Trains **all 36 models**. Re-check the printed estimate above and adjust `PER_CELL_STEPS` before running. Each cell trains one Mesa (or GDN) model on the mixed-distractor distribution and scores it at the fixed `n_distractors`.""")

code(r"""rows = mqar_delta_cg_sweep(
    DELTA_GRID, CG_GRID, mqcfg,
    n_pairs=NPAIRS, gap=GAP, n_distractors=N_DISTRACTORS, seed=SEED, with_gdn=True,
)

# compact table: Mesa accuracy, rows = CG, cols = δ
mesa = [r for r in rows if r["layer"] == "mesa"]
cgs  = sorted({r["cg_steps"] for r in mesa})
print(f"Mesa accuracy  (n_pairs={NPAIRS}, gap={GAP}, n_distractors={N_DISTRACTORS}):")
print("   CG \\ δ   " + "  ".join(f"{d:>5g}" for d in DELTA_GRID))
for cg in cgs:
    line = {r["delta"]: r["acc"] for r in mesa if r["cg_steps"] == cg}
    print(f"   {cg:>5}   " + "  ".join(f"{line[d]:5.2f}" for d in DELTA_GRID))
gdn = {r["delta"]: r["acc"] for r in rows if r["layer"] == "gated_deltanet"}
print("   GDN     " + "  ".join(f"{gdn[d]:5.2f}" for d in DELTA_GRID))""")

md(r"""## Headline — the (δ, CG) accuracy heatmap

The **shape** is the result. Flat-in-δ rows (each CG row roughly constant across δ) ⇒ **separable**. A tilt — the bright region's δ-location shifting as CG grows ⇒ **interactive** (δ* depends on CG). The GDN best-δ reference is annotated for scale.""")

code(r"""ax = plot_delta_cg_heatmap(rows,
        title=f"Mesa recall accuracy over (δ, CG)  —  distractor-MQAR (n_pairs={NPAIRS}, distractors={N_DISTRACTORS})")
plt.tight_layout(); plt.show()""")

md(r"""## Supplementary line plots

Same data, two readable slices (consistent palette). **Left question:** does the optimal δ shift with CG? **Right question:** does the CG plateau height depend on δ?""")

code(r"""# δ-curves: one line per CG (x = forget rate 1−δ, log; left = aggressive forgetting)
ax = plot_mse_vs(mesa, "forget_rate",
                 title="δ-curves: accuracy vs forget rate, one line per CG",
                 xlabel="forget rate  1−δ  (log; left = aggressive forgetting)",
                 ykey="acc", ylabel="answer-token exact-match accuracy",
                 logy=False, logx=True, lw=1.2, dim_alpha=1.0)
ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=8, title="Mesa CG")
plt.tight_layout(); plt.show()""")

code(r"""# CG-curves: one line per δ (relabel so plot_mse_vs groups by δ)
cg_view = [{**r, "label": f"δ={r['delta']:g}", "cg_steps_x": r["cg_steps"]} for r in mesa]
ax = plot_mse_vs(cg_view, "cg_steps_x",
                 title="CG-curves: accuracy vs CG steps, one line per δ",
                 xlabel="Mesa CG steps  (log)",
                 ykey="acc", ylabel="answer-token exact-match accuracy",
                 logy=False, logx=True, lw=1.2, dim_alpha=1.0)
ax.set_xticks(CG_GRID); ax.set_xticklabels([str(c) for c in CG_GRID])
ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=8, title="δ")
plt.tight_layout(); plt.show()""")

md(r"""### CG=1 caveat

From 03 and 04, **CG=1 (the bare GLA-like read-out) is decay-sensitive and seed-variable** — its row of the heatmap may look chaotic across δ. Anchor the separable-vs-interactive conclusion on the **CG ≥ 2** rows; do not over-interpret the CG=1 slice.""")

md(r"""## 5 · Verdict — separable or interactive? (computed)

Run-agnostic readout: for each CG≥2 row, the δ that maximizes accuracy (δ*). If δ* is ~constant across CG → **separable (A)**; if δ* shifts with CG → **interactive (B)**.""")

code(r"""def best_delta_per_cg(mesa_rows, cg_min=2):
    out = {}
    for cg in sorted({r["cg_steps"] for r in mesa_rows}):
        if cg < cg_min:
            continue
        cells = [r for r in mesa_rows if r["cg_steps"] == cg]
        best = max(cells, key=lambda r: r["acc"])
        out[cg] = (best["delta"], best["acc"])
    return out

bd = best_delta_per_cg(mesa, cg_min=2)
print("argmax-δ per CG (CG≥2, CG=1 excluded as unreliable):")
for cg, (d, a) in bd.items():
    print(f"   CG={cg:>2}:  δ* = {d:g}   (acc {a:.2f})")
deltas_star = [d for d, _ in bd.values()]
spread = max(deltas_star) - min(deltas_star)
print(f"\nδ* spread across CG = {spread:g}")
print("verdict:", "SEPARABLE (A) — δ* ~constant in CG" if spread <= 0.05 else
      "INTERACTIVE (B) — δ* shifts with CG")
# decay-sensitivity per CG: accuracy drop from best-δ to δ=0.5 (aggressive forgetting)
print("\ndecay sensitivity (best-δ acc − δ=0.5 acc), per CG:")
for cg in sorted({r["cg_steps"] for r in mesa}):
    cells = {r["delta"]: r["acc"] for r in mesa if r["cg_steps"] == cg}
    drop = max(cells.values()) - cells.get(0.5, float("nan"))
    print(f"   CG={cg:>2}:  Δ = {drop:+.2f}   ({'robust' if drop < 0.1 else 'sensitive'})")""")

md(r"""## Closing — which hypothesis won?

*(Fill against the heatmap + verdict cell. Anchored on CG ≥ 2; forgetting and CG are the only things varied, matched-init throughout.)*

- **If separable (A):** the optimal δ is ~constant across CG — solve quality and retention are **independent levers**, and the three-lever framing (CG, δ, capacity) is clean: each can be tuned without regard to the others on this task.
- **If interactive (B):** δ* shifts with CG — describe the direction. *Higher CG → lower δ\* needed* would mean the exact solve **substitutes** for retention (it can reconstruct targets from a partially-decayed state, so it needs to keep less), making the two levers partial substitutes. *Higher CG → higher δ\* needed* would mean they **compose** (you need both retention and a deep solve to clear interference).

**Tie back to the 04 hint.** 04 showed, at this very setting, that **CG=30 was decay-robust** (accuracy ~flat as δ fell) while **CG=1 was decay-sensitive** (collapsed as δ fell). The "decay sensitivity per CG" line above is exactly that hint measured across the full CG axis: if sensitivity *decreases* as CG grows, the full surface **generalizes** the one-slice observation — and that monotone trend *is* an interaction (more solve depth buys decay-robustness), i.e. evidence for **B** in the "substitute" direction. The headline heatmap's tilt (or lack of it) is the decisive picture.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
nbf.write(nb, "notebooks/05_delta_cg_interaction.ipynb")
print(f"wrote notebooks/05_delta_cg_interaction.ipynb with {len(cells)} cells")
