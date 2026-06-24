"""De-AI notebook 02: remove em dashes and AI tells; keep tables/numbers; fix code."""
import nbformat as nbf

P = "notebooks/02_noisy_drifting_regression.ipynb"
nb = nbf.read(P, as_version=4); C = nb.cells

def setmd(i, must, s):
    assert C[i].cell_type == "markdown" and must in C[i].source, f"cell {i}: {C[i].source[:60]!r}"
    C[i].source = s

setmd(0, "the honest comparison",
"""# 02 · Noisy and drifting regression (the honest comparison)

Notebook 01 used a clean, fixed task. Real regression is noisy (labels are corrupted) and non-stationary (the target map W moves over time). This notebook pushes MesaNet vs GDN along those two axes and asks where Mesa's exact solve is worth its cost.

What we expect, stated up front:

- (a) Noise makes Mesa's lead grow. Mesa's exact solve averages over all the examples, which cancels random label noise; GDN's single-step update does not. So Mesa's lead should widen as noise rises.
- (b) Drift makes Mesa's lead shrink or flip. When the target moves, fitting all of history is wrong, because old examples are stale. A method that down-weights old data should win, so Mesa's lead should shrink or flip as drift rises.

Forgetting is fixed here. Both layers keep their memory at the same near-1 rate, so forgetting is not a variable in this notebook. These sweeps isolate noise, drift, and CG compute. Sweeping the forget rate itself comes later.""")

setmd(1, "How the experiment is set up",
"""## How the experiment is set up

- One model per setting. For each (layer, CG, noise, drift) we train a small model at a fixed number of examples and score held-out batches. The swept axes here are noise and drift. All the sweep and plot code is in `compare/experiments.py`.
- Same scoring, via `synthtasks.metrics.mse_on_queries`.
- Forgetting matched near 1 (as in 01): both layers start by keeping about 98% of their memory per step (MESA_RET=4.0 gives decay about 0.982; GDN pinned to the same via `gdn_retention_init`). Without this they would forget at different rates, the confound from 01. Both stay trainable.
- Small, seeded, GPU. Tiny 2-layer models; the fla kernels need CUDA and Triton.""")

setmd(3, "what noise & drift do to the data",
"""## The difficulty knobs: what noise and drift do to the data

A look at the data (no model): noise pushes the target values off the clean line; drift slowly rotates the true map W across the sequence.""")

setmd(8, "Noise sweep (drift = 0)",
"""## Noise sweep (drift = 0)

Fix drift = 0 and vary the label-noise level σ. One model per (CG, σ), plus GDN. We show every Mesa CG curve (1, 2, 5, 10, 30) and GDN together on a log-y axis (the σ=1.0 errors are about 3x the rest, so log-y keeps both ends readable).""")

setmd(11, "Hypothesis (a): the Mesa curves should sit below",
"""Hypothesis (a) predicts the Mesa curves sit below GDN, and that the best Mesa (high CG) pulls further below GDN as σ grows. You can also read off how many CG steps it takes to beat GDN at each noise level.""")

setmd(12, "Value of CG steps as noise grows",
"""### Value of CG steps as noise grows

The same data the other way round: error vs Mesa CG steps, one curve per noise level (GDN is the same-colour dashed line). The y-axis is cut, because the σ=1.0 curves sit about 3x above the rest: they go in a small top panel while the σ up to 0.5 detail spreads out below. That way you can see how much each extra CG step buys at every noise level.""")

setmd(14, "If exact averaging pays off",
"""If exact averaging pays off under noise, the higher-noise curves should drop more across CG steps and sit further below their GDN line. In other words, extra CG compute is worth more when there is more noise.""")

setmd(15, "Drift sweep (noise = 0.25)",
"""## Drift sweep (noise = 0.25)

Hold a moderate noise and vary the drift rate. Forgetting is still fixed near 1 for both layers, so this asks what the exact solve does under drift when neither model is allowed to forget.""")

setmd(18, "Hypothesis (b): Mesa's lead over GDN",
"""Hypothesis (b) predicts Mesa's lead over GDN shrinks or flips as drift grows. Caveat: with forgetting fixed near 1, neither model can adapt, so any flip here is about the solve, not the forget gate (which the next sections sweep).""")

setmd(19, "Value of CG steps as drift grows",
"""### Value of CG steps as drift grows

Error vs Mesa CG steps, one curve per drift level (GDN dashed, same colour). Under drift, more CG means fitting stale history more exactly, so extra compute may stop helping, or even hurt.""")

setmd(21, "If drift kills the value",
"""If drift kills the value of exact fitting, the high-drift curves should flatten (CG steps stop buying error) and approach or cross their GDN line.""")

setmd(22, "did the hypotheses hold",
"""## Closing: did the hypotheses hold?

(Read off the runs above. Small-scale, forgetting fixed near 1 for both layers. "Mesa" means CG=30 unless noted.)

- (a) Noise, Mesa's lead grows: supported, but it saturates. Mesa stays at or below GDN at every σ, and the gap widens through the low-to-moderate range (about 0.03 at σ=0, about 0.06 at σ=0.5). At σ=1.0 both hit the large noise floor and the gap nearly closes. On the CG dial, 2 or more CG steps already beat GDN at every noise level.
- (b) Drift, Mesa's lead shrinks or flips: supported. At fixed noise 0.25, Mesa's margin shrinks as drift grows (0.053 down to 0.030) and flips at drift = 0.2, where GDN edges ahead (0.890 vs 0.902). This happens with forgetting fixed near 1, so it is the exact solve clinging to stale history, not a forgetting advantage. Letting Mesa forget should help, which the next sections test.
- Value of CG, noise vs drift. At drift = 0, going from CG=1 to CG=30 buys about 0.15 and Mesa beats GDN from CG=2 on. At drift = 0.2 the same dial buys only about 0.03 and every CG setting sits above GDN, so extra exact-fitting compute is mostly wasted once the target moves.

Because forgetting is fixed and equal, none of this is contaminated by the gate-init confound from 01. The drift flip is what the next sections chase: let the forget rate vary and see if it helps.""")

setmd(23, "Next: sweep the forget gate",
"""## Next: sweep the forget gate

The sweeps above fixed forgetting near 1. The drift flip in (b) hints that the real lever is the forget rate δ (1 means never forget, small means forget fast). To study it fairly, set both layers to the same δ:
- Mesa: `mesa_retention_init = logit(δ)` (the a_proj.bias).
- GDN: `gdn_retention_init = δ`.

Both are already wired through the config. The plan: sweep δ in {0.5, 0.8, 0.95, 0.99, 0.999}, focus on drift (where forgetting should matter), and expect a U-shape. Too much forgetting loses evidence, too little clings to a stale W, and the best δ should get smaller as drift rises. The next cells do exactly this.""")

setmd(24, "Forgetting sweep (noise = 0.25, drift = 0.1)",
"""## Forgetting sweep (noise = 0.25, drift = 0.1)

Sweep the keep-rate δ in {0.5, 0.8, 0.95, 0.99, 0.999} for both layers at one moderate-drift point. The question: with forgetting free to vary, can the right δ let GDN close the gap on Mesa (CG=30)?

- Mesa: `mesa_retention_init = logit(δ)`; GDN: `gdn_retention_init = δ`. Both stay trainable.
- The x-axis is the forget rate 1-δ on a log scale (left = forget fast, right = keep everything).""")

setmd(27, "Reading it (noise = 0.25, drift = 0.1)",
"""Results at noise = 0.25, drift = 0.1:

| forget rate 1-δ | 0.001 | 0.01 | 0.05 | 0.2 | 0.5 |
|---|---|---|---|---|---|
| Mesa (CG=30) | 0.701 | **0.653** | 0.655 | 0.661 | 0.687 |
| Gated DeltaNet | **0.676** | 0.679 | 0.705 | 0.724 | 0.718 |

Two findings, one against my own guess:

- Mesa is U-shaped: a little forgetting helps. Its best point is δ=0.99 (forget rate 0.01), error about 0.653, better than keeping everything (0.701). Under drift, dropping the stalest evidence improves the fit. The fixed-forgetting sweeps could not show this.
- GDN just wants to keep everything. Its best point is the most retentive one, and it gets worse as it forgets faster. So my guess that GDN would want to forget more was wrong: forgetting only hurts it here.
- Verdict: tuning the gate does not let GDN catch the exact solve. At each model's own best δ, Mesa (0.653) still beats GDN (0.676). Exact solve plus a little forgetting wins.

(Single seed, tiny models, so read this as a direction, not exact numbers. Next: δ by CG.)""")

setmd(28, "Do we also need to sweep CG",
"""### Do we also need to sweep CG here?

Not for this question. We fix Mesa at CG=30 (its strongest setting) on purpose, to ask one clean thing: given Mesa at full strength, can tuning the forget gate let cheap GDN catch up? A CG axis would only re-confirm the earlier finding that CG=2 and up already capture most of the benefit. The full δ-by-CG surface is worth it only if you care whether the best δ depends on CG, which is the δ-by-CG experiment (now notebook 03, section C).""")

setmd(29, "Forgetting sweep at stronger drift",
"""## Forgetting sweep at stronger drift (drift = 0.2)

Why again, and why at drift = 0.2. The first sweep (drift = 0.1) only showed forgetting helps Mesa a little. drift = 0.2 is the decisive point: it is where, with forgetting fixed, GDN actually overtook Mesa (about 0.890 vs 0.902). So we re-run the δ sweep right there.

The question: is that flip a real property of the exact solve, or just an artifact of forcing δ near 1? If letting Mesa forget moves its best point to a faster forget rate and pulls its error back to or below GDN, the flip was just clinging to stale history, fixable by the forget gate Mesa already has, and the best δ should fall as drift rises. Both layers share the same δ; Mesa stays at CG=30, so this is purely about forgetting.""")

setmd(32, "forgetting is the drift lever",
"""Forgetting is the drift lever, and it scales with drift.

- Mesa's best point forgets more than before. Its U-shape now bottoms out at δ about 0.95 (forget rate about 0.05), versus δ about 0.99 at drift = 0.1. So the best forget rate rises with drift, exactly as predicted: a faster-moving target wants more aggressive forgetting.
- Tuning δ erases the flip. At the fixed δ about 0.982 used elsewhere, GDN had overtaken Mesa at drift = 0.2 (about 0.890 vs 0.902). Letting Mesa forget pulls it back to about 0.895, a tie with GDN's best (about 0.893). So the flip was an artifact of clinging to stale history.

But the payoff shrinks: at drift = 0.1 tuning δ left Mesa clearly ahead (+0.023); at drift = 0.2 it only reaches a tie. Forgetting buys Mesa back into contention, but stronger drift still erodes its margin. (GDN stays monotone, it wants maximal retention. Single seed; directional.)""")

setmd(33, "Regularization sweep under noise",
"""## Regularization sweep under noise (Λ at noise = 1.0)

Why Λ, and why at high noise. Drift and noise break the exact solve differently. Drift is about stale data, so the fix is forgetting (above). Noise is about variance: the exact solve fits whatever labels it is given, so corrupted labels make it over-fit. The textbook fix is ridge regularization, shrinking the solution, and the best ridge strength should grow with noise. Mesa's ridge is exactly its Λ, which we froze into a sweepable knob in notebook 01b.

01b swept Λ on a clean task and found more Λ only hurt. noise = 1.0 is the opposite test: if the bias-variance story holds, an interior best Λ should appear here. Forgetting stays fixed near 1 and Mesa stays at CG=30, so Λ is the only thing changing.""")

setmd(36, "ridge is not the noise lever",
"""Ridge is not the noise lever here. Mesa's error stays about 1.6 across the whole Λ range and tracks GDN (about 1.61); larger Λ only makes it slightly worse. So tuning the ridge does not rescue Mesa under heavy noise at this setting.

Why nothing happens:
- Floor-dominated. With n=32, far above d=8, the solve is already low-variance, so the irreducible noise floor (about σ² = 1) dominates. There is little over-fitting for ridge to remove.
- The model trains around the frozen Λ. The other weights rescale to absorb it, so error barely depends on Λ until Λ is big enough to force the GLA collapse from 01b.

Honest takeaway: ridge only helps in the variance-dominated regime, when n is close to d, not when n is far above d. The next cell tests that.""")

setmd(37, "does ridge pay off as n → d",
"""## Follow-up: Λ by n_examples, does ridge pay off as n approaches d?

The Λ-at-noise=1.0 sweep above did nothing, because at n=32 (far above d=8) the error is floor-dominated. That makes a testable prediction: ridge should start helping only as n shrinks toward d.

This cell tests it: fix heavy noise (σ=1.0) and sweep Λ by n_examples, with n from 8 (=d) up to 48. Prediction:
- Left panel (error vs Λ per n): flat at large n, turning into a U-shape with an interior best Λ as n approaches d.
- Right panel (gain from tuning Λ): about 0 at large n, growing as n approaches d.

If the gain stays about 0 even at n = d, that means the trained model already absorbs the ridge and Λ just is not a useful lever at this scale, which is also a clean result.

> Cost: the most expensive cell here (about 42 small trainings). Trim the grids for a quick look.""")

setmd(40, "the prediction is refuted",
"""The prediction is refuted; under noise the lever is data, not ridge.

| n | best Λ | best Mesa | GDN | winner |
|---|---|---|---|---|
| 8 (=d) | 0.25 | 2.021 | **1.955** | GDN |
| 12 | 0.25 | 1.980 | **1.857** | GDN |
| 16 | 1 | 1.832 | **1.819** | GDN |
| 24 | 1 | 1.709 | **1.685** | GDN |
| 32 | 4 | **1.595** | 1.607 | Mesa |
| 48 | 1 | **1.450** | 1.497 | Mesa |

- Tuning Λ buys about 0 at every n, and does not grow as n approaches d. At n=8 and n=12 the best Λ is the smallest one (gain exactly 0); the tiny bumps sit at larger n and are within single-seed noise. No interior best Λ appears. The bias-variance prediction fails at this scale: the trained weights absorb the frozen Λ, and at σ=1.0 even n=d is floor-dominated.
- The real signal is a data crossover. Reading GDN vs best-Mesa down the table: GDN wins under heavy noise while data is scarce (n up to 24) and only loses once n is around 4d (n=32, 48). So Mesa's noise-averaging edge needs enough examples to show up; when data is scarce and noisy, the cheap delta rule is more robust.

Closing the loop: 01b showed Λ matters on clean data; here under heavy noise Λ is inert and the winner is set by how much in-context data there is, not the ridge. Contrast with drift, where forgetting was a real lever. (Single seed, tiny models; directional.)""")

for c in C:
    if c.cell_type == "code" and "—" in c.source:
        c.source = c.source.replace("—", "-")
for c in C:
    if "—" in c.source:
        c.source = c.source.replace(" — ", ", ").replace("—", ", ")
assert not any("—" in c.source for c in C), "em dash survived"
nbf.write(nb, P)
print(f"de-AI'd 02 ({len(C)} cells); em dashes remaining:",
      sum(c.source.count("—") for c in C))
