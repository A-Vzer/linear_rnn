"""De-AI notebook 01: remove em dashes and AI tells from prose; fix code strings."""
import nbformat as nbf

P = "notebooks/01_regression_sanity.ipynb"
nb = nbf.read(P, as_version=4); C = nb.cells

def setmd(i, must, s):
    assert C[i].cell_type == "markdown" and must in C[i].source, f"cell {i}: {C[i].source[:60]!r}"
    C[i].source = s

setmd(0, "big question this project asks",
"""# 01 · In-context linear regression (sanity check)

The question this project asks: going from cheap gated linear attention up to MesaNet's exact solve, when is the extra test-time compute worth more than Gated DeltaNet's single-step rule?

This notebook is the smallest first step, a sanity check on a clean, easy version of the task. It is not a result. The goal is to reproduce what the papers already imply and to confirm our setup is correct before trusting it on harder tasks.

What we expect. MesaNet fits all of the in-context examples at once (an exact least-squares solve), so it should reach low error with few examples (about the input dimension d). Gated DeltaNet makes one small update per token, so it should need many more examples to match. We expect Mesa's error curve to sit to the left of GDN's.

If that gap does not show up, that tells us something about our setup, not that the papers are wrong. Hence the name: sanity check.""")

setmd(1, "How the experiment is set up",
"""## How the experiment is set up

- Train once, test at every difficulty. We train one model across a range of context lengths (the number of in-context examples), then measure held-out error separately at each length. This matches how the layer is actually used: one model, many context lengths.
- Easy task. Noiseless y = Wx with a fixed W. Noise and a moving W come in notebook 02.
- Small and reproducible. 2-layer models, all seeded. Held-out batches use fresh seeds, so the model has to generalize, not memorize.
- Same scoring everywhere, via `synthtasks.metrics.mse_on_queries`, so the model's error and the closed-form best-possible floor are directly comparable.
- Fair starting point for forgetting (disclosed). Out of the box, MesaNet starts by forgetting about half its memory each step, while GDN starts by keeping almost all of it. This task needs every example kept, so the stock Mesa setting is a bad start and makes it look worse than it is. We start both layers from the same keep-almost-everything setting (`MESA_RETENTION`; still trainable). Set it to `None` to see the stock behaviour: GDN looks dominant until about 5x more training, which is a setup artifact, not a real finding.

> Needs a GPU. MesaNet and GatedDeltaNet are flash-linear-attention layers that need CUDA and Triton. Only the CG check (Step 1) runs on CPU.""")

setmd(3, "what the model sees",
"""## The data: what the model sees

A quick look at the generator (no model yet): the (x, y) examples laid out as a token stream, and the target value at each query position the model has to predict.""")

setmd(7, "what does the CG dial actually do",
"""## Step 1: a free correctness check on the CG dial

Before comparing anything, we check Mesa's CG dial against the installed kernel, not our memory of the paper. MesaNet produces its answer by solving a small linear system with conjugate gradient (CG); the number of CG steps k is the compute dial.

People often say "k=0 is just gated linear attention (GLA)." In the installed `fla` kernel that is not true. The solver starts from zero, so:

- k=0 gives an output of exactly zero (no mixing at all), a useless floor, not GLA.
- k=1 gives the GLA read-out (up to a per-token scale). This is the real cheap-GLA point.
- large k gives the exact solve.

So the cheap end of the dial here is k=1, and we drop the useless k=0. The check below uses fla's plain-PyTorch reference (CPU, no Triton) and must PASS before we trust anything.""")

setmd(9, "a second way to collapse to GLA",
"""## Step 1b: the Λ knob, a second way to collapse to GLA, and what it costs

The CG count is one way to dial Mesa down to the cheap GLA read-out. The ridge term Λ is a second, independent one. Mesa solves x = (H + Λ)⁻¹ q. When Λ is large, (H + Λ)⁻¹ is about 1/Λ, the accumulated history H drops out, and the read-out lands on the same GLA reference as before, up to a constant scale. Two independent knobs (CG toward 1, and Λ large) landing on the same point is strong evidence the code is correct. The check below confirms this at Λ=50 on CPU and must PASS.

The plot shows what that collapse costs. Leaning on Λ throws away the in-context history the exact solve needs. We freeze Λ at a range of values (CG fixed at 30) and plot held-out error vs Λ, with trained GDN as the reference. Error rises as Λ grows: Mesa beats GDN only at small Λ, and by the paper's Λ=50 it has degraded past GDN. This is the Λ version of the CG sweep in Step 2.""")

setmd(12, "from cheap (GLA-like) to exact",
"""## Step 2: Mesa CG sweep, from cheap (GLA-like) to exact

Now hold the task fixed and sweep Mesa's compute dial k in {1, 2, 5, 10, 30}. From Step 1, k=1 is the cheap GLA point and larger k approaches the exact solve. GDN is the reference, the cheap incumbent we are asking whether the extra Mesa compute can beat. Mesa at k=30 against the GDN reference is the main comparison.

We expect the curves to drop as k grows (more compute, lower error at fewer examples), with diminishing returns, and the k=1 curve to be the weakest.

> Note: GDN's error also depends on context length, so we plot its full curve, not a flat line.""")

setmd(15, "More CG steps = more compute",
"""More CG steps means more compute, which means closer to the exact solve, so the curves should drop as k grows and level off by about k=10. The gap between k=1 (cheap) and k=30 (exact) is the value of the extra compute. The gap between the Mesa curves and the GDN reference is the trade-off this project is about.""")

# --- code cells: drop em dashes (use a hyphen; safe in comments/strings) ---
for c in C:
    if c.cell_type == "code" and "—" in c.source:
        c.source = c.source.replace("—", "-")

# --- backstop: no em dash may remain anywhere ---
for c in C:
    if "—" in c.source:
        c.source = c.source.replace(" — ", ", ").replace("—", ", ")
assert not any("—" in c.source for c in C), "em dash survived"
nbf.write(nb, P)
print(f"de-AI'd 01 ({len(C)} cells); em dashes remaining:",
      sum(c.source.count("—") for c in C))
