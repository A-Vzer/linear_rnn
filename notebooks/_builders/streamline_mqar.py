"""Streamline notebooks/03_mqar.ipynb (markdown/structure only; no execution):
  - add one shared 'Conventions' cell; trim the 3 repeated design cells
  - unique section-prefixed sub-headers (A1.., B1.., C1..); de-dup caveats
  - strip the redundant import block from the 3 section config cells
Run from repo root.
"""
import nbformat as nbf

P = "notebooks/03_mqar.ipynb"
nb = nbf.read(P, as_version=4)
C = nb.cells

def set_first_line(i, new):
    lines = C[i].source.split("\n")
    lines[0] = new
    C[i].source = "\n".join(lines)

def set_src(i, s):
    C[i].source = s

def strip_imports_keep_config(i, marker="# ----- config"):
    s = C[i].source
    if marker in s:
        C[i].source = s[s.index(marker):]

# --- sanity: confirm we're editing the cells we think we are ---
assert C[9].source.startswith("## Experimental design"), C[9].source[:40]
assert C[37].source.startswith("## Experimental design")
assert C[49].source.startswith("## Design")
assert C[35].source.startswith("## Closing")
assert C[47].source.startswith("## Closing")
assert C[61].source.startswith("## Closing")

# --- trim the 3 design cells to section-specific notes (shared parts -> Conventions) ---
set_src(9, (
    "### A · Design\n\n"
    "Capacity sweep + a control. Train one model per `(layer, CG)` on a mix of `n_pairs` × `gap` "
    "(the two-axis mix is why `train_mqar` exists, vs the single-knob `train_across_eval`); evaluate "
    "per setting. Forgetting matched near 1 (δ≈0.982). See **Conventions** above."
))
set_src(37, (
    "### B · Design\n\n"
    "Distractors are never-queried key→value pairs (keys disjoint from the targets) written *after* "
    "the targets, via `make_mqar(..., n_distractors=k)`; training mixes `n_distractors` "
    "(`MQARConfig.train_distractors`). **Clean baseline:** `n_pairs` fixed small (recall ≈ perfect at "
    "0 distractors) and `gap` small, so any drop is attributable to interference. §B1 holds forgetting "
    "matched near 1; §B2 *sweeps* δ (retraining per δ). See **Conventions**."
))
set_src(49, (
    "### C · Design\n\n"
    "The joint **δ × CG** surface on the fixed §B distractor slice (`n_pairs`, `gap`, `n_distractors` "
    "reused from B, where CG=30/CG=1 separated). **One model per (δ, CG) cell:** 6 δ × 5 CG = 30 Mesa "
    "+ 6 GDN refs = 36 trainings — *the most expensive experiment in the suite*; the config cell prints "
    "the cell count + wall-clock estimate so the per-cell budget is chosen before the sweep. All via "
    "`mqar_delta_cg_sweep`. See **Conventions**."
))

# --- de-duplicate caveats in the closings (point to Conventions; keep section-specific bits) ---
A = C[35].source
A = A.replace(
    "**Caveats.** (i) Mesa's exact-solve path is optimisation-sensitive (grad-clip + LR warmup "
    "required; without them the high-CG models collapse). (ii) GPU/Triton kernels are not "
    "bit-deterministic and the **decay-sensitive** CG=1 (and to a lesser degree GDN) are additionally "
    "seed-variable; the **stable findings are: the CG≥2 plateau ≫ GDN at high load, and CG=30 is the "
    "stable, decay-robust operator** — finer rankings (and the CG=1 point in particular) sit inside the noise.",
    "**Stable findings:** the CG≥2 plateau ≫ GDN at high load, and CG=30 is the stable, decay-robust "
    "operator; the CG=1 point sits inside the noise. *(Reproducibility & stability caveats: see Conventions.)*")
C[35].source = A

B = C[47].source
B = B.replace(
    "**Caveats.** GDN (and CG=1) recall is run-variable at this scale — Mesa CG=30 is the stable "
    "operator; absolute GDN levels in §1 wobble seed-to-seed, but its position below the "
    "distractor-immune CG=30, and the monotone retention story in §2, are stable. Distractors here use "
    "**disjoint** keys (capacity/recency interference); colliding-key distractors would be a harsher, "
    "separate test.",
    "**Caveat (task-specific).** Distractors here use **disjoint** keys (capacity/recency interference); "
    "colliding-key distractors would be a harsher, separate test. *(Reproducibility & stability caveats: "
    "see Conventions.)*")
C[47].source = B

# --- unique, section-prefixed sub-headers (demote H2 sub-sections to H3) ---
set_first_line(11, "### A · Train once (then sweeps are eval-only)")
set_first_line(13, "### A1 · Capacity sweep — accuracy vs `n_pairs` (gap fixed)")
set_first_line(17, "### A1b · Capacity control — capacity, or forgetting?")
set_first_line(21, "### A2 · Gap sweep — accuracy vs `gap`")
set_first_line(25, "### A3 · CG cost-vs-quality (headline)")
set_first_line(29, "### A4 · Depth (position-resolved)")
set_first_line(33, "### A5 · Summary (computed)")
set_first_line(35, "### A · Closing — did the hypotheses hold?")
set_first_line(39, "### B1 · Distractor sweep")
set_first_line(43, "### B2 · Forget gate × distractors")
set_first_line(47, "### B · Closing — distractors & the forget gate")
set_first_line(51, "### C · Run the sweep (expensive)")
set_first_line(53, "### C1 · Heatmap (headline)")
set_first_line(55, "### C2 · Line plots")
set_first_line(58, "### C · CG=1 caveat")
set_first_line(59, "### C3 · Verdict (computed)")
set_first_line(61, "### C · Closing — which hypothesis won?")

# --- strip redundant import blocks from section config cells (header imports everything) ---
for i in (10, 38, 50):
    strip_imports_keep_config(i)
# §1b inline import (header already has plot_capacity_forget_control)
C[18].source = "\n".join(l for l in C[18].source.split("\n")
                         if "import plot_capacity_forget_control" not in l).lstrip("\n")

# --- insert one shared Conventions cell before §A (index 8) ---
conv = nbf.v4.new_markdown_cell(
    "## Conventions (shared across A, B, C)\n\n"
    "All three experiments use the same harness and reporting conventions; stated once here.\n\n"
    "- **Train-across / evaluate-per-setting.** Each section trains its *own* models (training "
    "distribution differs per experiment), then evaluates the frozen model at each probed setting. "
    "**One model per `(layer, CG, δ)` configuration** — they can't share weights (gate init and solve "
    "depth differ structurally).\n"
    "- **Matched forgetting init.** Both layers pinned to the *same* initial per-step decay δ so the "
    "comparison isolates capacity/algorithm, not the gate prior: Mesa `mesa_retention_init = logit(δ)` "
    "(a_proj.bias), GDN `gdn_retention_init = δ` (`A_log=0`, `dt_bias`). Both stay trainable. (Verified "
    "against fla in 02.)\n"
    "- **Test-time compute = CG steps.** CG=1 ≈ the GLA read-out, CG→30 ≈ the exact $(H+\\lambda I)^{-1}q$ "
    "solve (CG semantics verified in 01). Analytic cost: **Mesa(CG=k) ≈ k× GLA mixer FLOPs; GDN ≈ 1× GLA**.\n"
    "- **Identical scoring** via `synthtasks.metrics.mqar_exact_match` (answer tokens only).\n"
    "- **Stabilised training.** Mesa's exact-solve path is optimisation-sensitive, so `MQARConfig` "
    "defaults to gradient clipping + LR warmup; without them the high-CG models occasionally collapse.\n"
    "- **Reproducibility caveat.** fla/Triton kernels are not bit-deterministic and **CG=1 (and, less so, "
    "GDN) are seed-variable** — absolute accuracies wobble run-to-run; the *qualitative orderings* are "
    "stable, the third decimal is not. Anchor conclusions on **CG ≥ 2**.\n"
    "- Small 2-layer models; GPU (CUDA + Triton) required."
)
C.insert(8, conv)

nbf.write(nb, P)
print(f"streamlined 03_mqar.ipynb -> {len(C)} cells")
