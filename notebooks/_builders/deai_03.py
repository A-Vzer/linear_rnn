"""De-AI notebook 03: remove em dashes and AI tells; fix headers and plot strings."""
import nbformat as nbf

P = "notebooks/03_mqar.ipynb"
nb = nbf.read(P, as_version=4); C = nb.cells

def setmd(i, must, s):
    assert C[i].cell_type == "markdown" and must in C[i].source, f"cell {i}: {C[i].source[:60]!r}"
    C[i].source = s

setmd(0, "recall, capacity",
"""# 03 · MQAR: recall, capacity, interference, and the δ × CG surface

Multi-query associative recall (MQAR) is the memory task where MesaNet's exact "fit all of history" solve is supposed to pay off. Three experiments:

- A · Capacity: recall vs how many key→value pairs you store (n_pairs), plus a control that separates "state full" from "forgot the early ones."
- B · Distractors: recall when extra never-asked pairs compete for the memory, and what the forget gate does under that interference.
- C · δ × CG: the joint surface of Mesa's two knobs (how much it keeps, and how hard it solves).

Each section trains its own models and stands alone; the data is shown inline below. (This merges the former notebooks 03, 04, and 05.)""")

setmd(3, "what the model sees",
"""## MQAR data: what the model sees

Three quick looks at the generator (no model): a small write→query→answer sample, the same with a heavy memory load (large n_pairs), and one with distractors (extra pairs that are never asked about).""")

setmd(8, "Conventions (shared",
"""## Conventions (shared across A, B, C)

The same setup and reporting apply to all three; stated once here.

- Train once, test at every setting. Each section trains its own models on a mix of difficulties, then tests the frozen model at each setting. One model per (layer, CG, δ): they cannot share weights, since the forget setting and the solve depth change the model structurally.
- Forgetting matched. Both layers start at the same keep-rate δ, so the comparison is about capacity and algorithm, not the starting forget bias: Mesa via `mesa_retention_init = logit(δ)`, GDN via `gdn_retention_init = δ`. Both stay trainable.
- Compute is CG steps. CG=1 is about the cheap GLA read-out, CG toward 30 is the exact solve. Cost: Mesa at CG=k is about k times the GLA cost; GDN is about 1 times.
- Same scoring, via `mqar_exact_match` (answer tokens only).
- Stabilised training. Mesa's exact solve is touchy to optimise, so training uses gradient clipping and LR warmup by default; without them the high-CG models sometimes blow up.
- Reproducibility. The GPU kernels are not bit-exact, and CG=1 (and, less so, GDN) wobble seed to seed. The orderings are stable, the third decimal is not. Trust CG=2 and up.
- Small 2-layer models; needs a GPU.""")

setmd(9, "where the exact solve should pay off",
"""## A · Capacity: where the exact solve should pay off

Notebooks 01 and 02 were regression. This is the recall test: write a set of key→value pairs into memory, then read them back. This is exactly where Mesa's "fit all of history" solve and its bigger state should earn their cost.

What we expect:

- (a) Easy when light. With few pairs, both models recall almost perfectly.
- (b) Mesa fades slower as load grows. As the number of pairs passes what the state can cleanly hold, GDN's accuracy should drop faster than Mesa's, because the exact solve can untangle overlapping keys that a single GDN step cannot.
- (c) Mesa's lead grows with load and with the write→query gap.

A confound to watch. Recall is where retention matters most, so the forget-gate starting bias (flagged in 01 and 02) matters most here. We therefore start both layers at the same keep-rate. If Mesa still lost here, that would signal something upstream is broken, since this is the task it should win.""")

setmd(10, "### A · Design",
"""### A · Design

Capacity sweep plus a control. Train one model per (layer, CG) on a mix of n_pairs and gap, then test per setting. Forgetting matched near 1 (δ about 0.982). See Conventions.""")

setmd(12, "Train once (sweeps are test-only",
"""### A · Train once (sweeps are test-only after this)

Train the full set, Mesa at CG in {1,2,5,10,30} plus GDN, six models, each on the mixed (n_pairs, gap) data. Training is the only slow step; every figure below reuses these frozen models. (Mesa's recall trains slowly, so this is the slowest cell.)""")

setmd(14, "A1 · Capacity sweep",
"""### A1 · Capacity sweep: accuracy vs n_pairs (gap fixed)

Fix the gap, vary n_pairs from clearly light (2) to clearly overloaded (96). Tests (a) tie when light and (b) GDN fades faster than Mesa.""")

setmd(17, "(a) holds:",
"""(a) holds: all curves start near 1.0 when light. (b) holds: as load grows into the shaded band, GDN falls fastest and the exact-solve Mesa (CG=30) holds up best. The big, reliable gap here is Mesa-class over GDN. The cheap Mesa (CG=1) sits in between and is the most run-variable curve; section A1b explains why. Unlike the exact solve, CG=1 is hurt by forgetting at this keep-rate, which also quietly muddies this plot. Read the CG=1 vs CG=30 gap only after A1b.""")

setmd(18, "A1b · Capacity control",
"""### A1b · Capacity control: is it "state full" or "forgot the early ones"?

A1 ran at keep-rate δ about 0.98. That sounds like "barely forgets," but it compounds: over a write block of 100-plus tokens, the earliest pairs fade a lot. The memory horizon is about 1/(1-δ), roughly 56 tokens, or 28 pairs. That sits right on top of the state's capacity limit (about head_dim = 32), so the A1 drop could be the state running out of room, or the gate forgetting early writes. The two are tangled together.

This control re-runs the sweep with forgetting basically off (δ toward 1, horizon about 3000 tokens). If a model's drop moves to higher n_pairs with forgetting off, its A1 drop was partly forgetting; if the curve does not move, it is real capacity.""")

setmd(21, "this is the key control",
"""This is the key control. The two settings cleanly separate the models:

- Mesa CG=30 (exact solve): the two curves sit on top of each other. The solve recovers each pair no matter how faded it is, so it is insensitive to the gate. Its curve is real capacity, not forgetting.
- Mesa CG=1: forgetting off lifts it a lot under load. The cheap read-out really does suffer the recency bias, so much of CG=1's A1 drop was forgetting, not a full state. (GDN can shift too, by a smaller, run-variable amount; the summary cell labels each model from its actual shift.)

So: (i) at δ about 0.98, A1 undersells the cheap read-out, and the exact-solve curve is the clean capacity probe; (ii) this is why CG=1 wobbles run to run; (iii) it also explains part of the headline CG gap, since with forgetting off the CG=30 vs CG=1 gap shrinks (some of it was a forgetting handicap, not better solving). The δ about 0.98 value comes from the regression notebooks, where sequences are short enough that this barely matters.""")

setmd(22, "A2 · Gap sweep",
"""### A2 · Gap sweep: accuracy vs gap

Hold a moderate n_pairs and vary the write→query gap (blank filler tokens between writing and asking). Tests the retention half of (c).""")

setmd(25, "If holding memory over time",
"""If holding memory over time were the bottleneck, accuracy would drop as the gap grows. But the gap is blank filler (no competing pairs), and forgetting is near 1, so a flat result is the honest expectation. The bottleneck is how many pairs (n_pairs), not distance. Any vertical gap between models is the same capacity gap as in A1.""")

setmd(26, "A3 · CG cost-vs-quality",
"""### A3 · CG cost vs quality (headline)

The figure the suite builds toward: at a load where capacity actually bites, how does accuracy move as we spend more compute (Mesa CG steps)? GDN is the flat reference (no CG dial, cost about 1 times GLA). The annotation spells out the cost: each CG step is about one more GLA-equivalent pass.""")

setmd(29, "the cost-vs-quality story",
"""The cost-vs-quality story, two findings, most reliable first.

1. CG=2 and up reach a plateau well above GDN. From CG=2 on, Mesa sits on a flat plateau above the GDN line. A couple of CG steps already capture the exact solve's untangling of overlapping keys (echoing 02's "CG=2 and up capture most of it"). So about 2 times the GLA cost (CG=2) already buys the plateau and clears GDN; more CG adds little. The gap from the plateau down to GDN is the exact solve's advantage over a single delta step.

2. CG=1 is the unreliable outlier; do not read it as "the cheap Mesa." The bare read-out at CG=1 is both forgetting-sensitive (A1b) and seed-variable, so it can land near the plateau or below GDN (as it does here). Removing its forgetting handicap (δ toward 1, A1b) shrinks the CG=30 vs CG=1 gap, but CG=1 is too noisy to trust a precise number, so we anchor on the CG=2-and-up plateau.

Note: an earlier higher-LR run showed CG=1 beating CG=30. That was training instability in the exact-solve path, not a real effect; gradient clipping and warmup fixed it. The lesson: stress-test any "more compute changes recall" claim against both the optimiser and the gate (A1b) first.""")

setmd(30, "A4 · Depth",
"""### A4 · Depth (per-query position)

A cheap depth probe with no extra length sweep: at one load, plot accuracy at each query position (1st query, 2nd, and so on). Later queries sit deeper past the writes, so this shows whether Mesa's edge grows with depth.""")

setmd(33, "flat across query position",
"""Mesa's accuracy is basically flat across query position (steady from first to last), while GDN drifts slightly down. So Mesa's relative edge grows only a little with depth, and that is GDN slipping, not Mesa rising. "Mesa's edge grows late" is weakly supported here: within the query block, depth is not the bottleneck, load is (matching the flat gap sweep). (Caveat: this bins by query position, which is roughly equal difficulty. The forgetting-sensitive version would bin by where the asked key was written, since early-written keys fade faster at δ below 1, which A1b already isolates.)""")

setmd(34, "A5 · Summary",
"""### A5 · Summary (computed)

A compact readout that does not depend on hardcoded numbers: the easy-regime tie, the Mesa vs GDN margin at high load, the best CG at the headline setting, and the depth trend.""")

setmd(36, "A · Closing",
"""### A · Closing: did the hypotheses hold?

(Read against the plots and tables above. Small-scale, forgetting matched near 1 for both.)

- (a) Easy-regime tie: held. When light, every model recalls almost perfectly.
- (b) Mesa fades slower than GDN: held, and this is the headline result. With forgetting matched (so it is not a starting-bias artifact), the exact-solve Mesa (CG=30, and the CG=2-and-up plateau) stays above GDN as load grows, the gap widening through the capacity band. That is the payoff of keeping the full state and reading it out with a least-squares solve instead of one delta step. Mesa winning here is the healthy outcome.
- (c) Edge grows with load and gap: refined. Grows clearly with load (b); flat in gap (blank filler), only mildly depth-dependent. On compute, the CG=2-and-up plateau sits well above GDN and about 2 steps already get there; CG=1 alone is an unreliable outlier, so we anchor on CG=2 and up.

The key control (A1b). The δ about 0.98 keep-rate (inherited from the regression notebooks) compounds into a recency bias whose reach (about 28 pairs) sits right at the state-capacity limit, tangling the two. Turning forgetting off untangles them: Mesa CG=30 does not move (real capacity), while CG=1 improves a lot (its drop was partly forgetting). So the exact-solve curve is the clean capacity probe. For a pure capacity claim, use CG=30 or δ toward 1.

On the confound. Both gates share the same keep-rate, so the Mesa vs GDN gap is not a forgetting artifact. Stock GDN (mixed per-head init, some heads forgetting fast) actually recalls worse than matched GDN, so matching helps GDN, making this the conservative setting for (b).

Stable findings: the CG=2-and-up plateau is well above GDN at high load, and CG=30 is the stable operator; the CG=1 point sits in the noise. (Reproducibility caveats: see Conventions.)""")

setmd(37, "B · Distractors",
"""## B · Distractors: interference and the forget gate

Section A found the recall bottleneck was capacity (n_pairs); distance (a blank gap) and depth were flat, because blank filler carries no competing information. Here we fill the gap with distractors: extra key→value pairs that are never asked about, with keys different from the targets, written after the targets so they are the most recent memories before the questions. That turns "distance" into real interference. And because the distractors are newer than the targets, forgetting now matters: a recency-biased state keeps the distractors and loses the older targets.

What we expect:

- (a) Distractors hurt recall, and GDN faster than Mesa. A single GDN step can be overwritten by recent competing writes; the exact solve untangles them, so Mesa (CG=30) should stay much flatter.
- (b) Forgetting now hurts recall, the mirror image of drift (02). Under drift, old data is stale, so forgetting helps. Here the thing you must keep (targets) is older than the interference (distractors), so forgetting throws away exactly what you need: accuracy should rise with retention (δ toward 1), with no sweet spot in the middle.

This is also the concrete answer to "where on recall should I sweep the forget gate?": here, since distractors are what finally give it teeth.""")

setmd(38, "### B · Design",
"""### B · Design

Distractors are never-asked key→value pairs (different keys from the targets) written after the targets, via `make_mqar(..., n_distractors=k)`; training mixes the count. Clean baseline: n_pairs and gap small, so any drop is due to the distractors. B1 keeps forgetting matched near 1; B2 sweeps δ (retraining per δ). See Conventions.""")

setmd(40, "B1 · Distractor sweep",
"""### B1 · Distractor sweep

Train Mesa (CG=1 and CG=30) and GDN with distractors mixed in, then vary the number of distractors at the fixed clean base. Tests (a): GDN should fall as interference grows; Mesa (CG=30) should stay flat.""")

setmd(43, "Mesa is basically immune",
"""Mesa is basically immune to distractors. CG=30 stays near the ceiling even with 64 never-asked competitors (CG=1 sits close behind when it trains well, as here). The exact solve untangles the different-key distractors, so they cost it almost nothing. GDN sits well below and slips a little, since a single delta step cannot separate the competing writes as cleanly. Notice the curves are fairly flat: at near-1 retention, the number of distractors barely matters (different keys get routed apart, not collided), so B1 is about levels (Mesa above GDN), not slopes. Distractors really bite only when the gate also forgets, which is B2. (CG=1 is the run-variable one; CG=30 is stable.)""")

setmd(44, "B2 · Forget gate × distractors",
"""### B2 · Forget gate × distractors

Now sweep the shared keep-rate δ for both layers at a fixed distractor load, retraining per δ. The x-axis is the forget rate 1-δ (left = forget fast, right = keep everything). Tests (b).""")

setmd(47, "the mirror image of drift",
"""The headline, and the mirror image of drift (02). Forgetting is disastrous for recall under distractors. With fast forgetting (δ=0.5, a memory of about 2 tokens) both layers collapse toward chance, because the state cannot hold the older targets past the newer distractors. As retention rises (δ toward 1), recall comes back steadily, with no sweet spot in the middle, the opposite of the drift sweep in 02, where a little forgetting helped. That contrast is the point: under drift the old data is stale, so forget it; in recall the thing to keep is older than the interference, so keep it.

Two more reads: (i) Mesa recovers faster than GDN as retention rises (the exact solve rebuilds the targets from a partly-faded state while GDN lags); (ii) so on recall the answer is keep, on drift it is forget.""")

setmd(48, "B · Closing",
"""### B · Closing: distractors and the forget gate

- (a) held, with a twist. Distractors barely dent the exact solve (Mesa CG=30 stays near the ceiling at every count), while GDN sits well below (CG=1 is run-variable). At near-1 retention the count barely matters, since different keys get routed apart, not collided, so B1 is about levels, not slopes.
- (b) held cleanly, and it is the headline. Forgetting hurts recall under distractors: fast forgetting (δ at or below 0.8) collapses both toward chance, and accuracy rises steadily with retention. This is the mirror image of drift (02), and the recall setting where the forget gate finally has teeth. Mesa recovers fastest.

Where to sweep forgetting, across the whole suite. Clean or blank-gap recall (section A): do not bother, keep everything. Drift (02): a real sweet spot in the middle. Distractor recall (here): keep everything, useful as the contrast to drift. Together: forget the stale, keep the interfered-against.

Caveat: distractors here use different keys from the targets; same-key distractors would be a harsher, separate test. (Reproducibility caveats: see Conventions.)""")

setmd(49, "do Mesa's two knobs interact",
"""## C · δ × CG: do Mesa's two knobs interact?

We have studied Mesa's two knobs separately: the forget gate δ (how much it keeps; 02 and section B) and the solve depth CG (how hard it solves; section A). This section maps their joint surface on the task where each one matters strongly, distractor recall (section B), and asks whether they are independent or interact.

Why this task: here both knobs work on the same problem, keeping the older targets readable through the newer distractors. δ matters a lot (B2) and CG has the plateau-vs-collapse shape (B1). Drift (02) has δ effects too, but probably independent ones; distractor recall is where they might genuinely interact.

What we expect:

- A, independent. The best δ is about the same at every CG: the two knobs do not talk to each other.
- B, interact. The best δ depends on CG. For example, more CG might mean you need less retention (the solve can rebuild a partly-faded state), or you might need both. Either way, a real finding about how Mesa's mechanisms combine.

Both outcomes are useful. It should also match the section B hint: there, CG=30 was forgetting-robust while CG=1 was forgetting-sensitive. This sweep asks whether that one slice holds across the whole (δ, CG) grid.""")

setmd(50, "### C · Design",
"""### C · Design

The joint δ-by-CG grid on the fixed section B distractor setting (same n_pairs, gap, n_distractors, where CG=30 and CG=1 separated). One model per (δ, CG) cell: 6 δ by 5 CG is 30 Mesa, plus 6 GDN, for 36 trainings. This is the most expensive experiment in the suite, so the config cell prints the cell count and a time estimate, letting you pick the per-cell budget before running. See Conventions.""")

setmd(52, "C · Run the sweep",
"""### C · Run the sweep (expensive)

Warning: trains all 36 models. Check the printed estimate above and adjust `PER_CELL_STEPS` before running. Each cell trains one model on the distractor data and scores it at the fixed distractor load.""")

setmd(54, "C1 · Heatmap",
"""### C1 · Heatmap (headline)

The shape is the result. If each CG row is roughly flat across δ, the knobs are independent. If the bright region shifts along δ as CG grows, they interact (best δ depends on CG). GDN's best-δ score is noted for scale.""")

setmd(56, "C2 · Line plots",
"""### C2 · Line plots

The same data, two readable slices. Left: does the best δ shift with CG? Right: does the CG plateau height depend on δ?""")

setmd(59, "C · CG=1 caveat",
"""### C · CG=1 caveat

From sections A and B, CG=1 is forgetting-sensitive and seed-variable, so its row may look noisy across δ. Base the independent-vs-interact call on the CG=2-and-up rows; do not over-read the CG=1 row.""")

setmd(60, "C3 · Verdict",
"""### C3 · Verdict (computed)

For each CG=2-and-up row, find the δ with the best accuracy (call it δ-star). If δ-star is about the same across CG, the knobs are independent. If δ-star shifts with CG, they interact.""")

setmd(62, "C · Closing",
"""### C · Closing: which way did it go?

(Read against the heatmap and the verdict cell; based on CG=2 and up.)

- If independent: the best δ is about the same at every CG. The two knobs do not talk, and the three-knob picture (CG, δ, capacity) is clean.
- If interact: the best δ shifts with CG. More CG meaning less retention needed would say the exact solve substitutes for keeping memory (it rebuilds the targets from a faded state). More CG meaning more retention needed would say they team up (you need both to clear interference).

Tie back to section B. Section B showed, at this setting, that CG=30 was forgetting-robust while CG=1 was forgetting-sensitive. The "decay sensitivity per CG" line above measures that across the full CG axis. If sensitivity falls as CG grows, the surface generalizes the section B slice, and that trend is itself an interaction (more solve depth buys forgetting-robustness), evidence for the interact case in the "substitute" direction. The heatmap's tilt, or flatness, is the decisive picture.""")

for c in C:
    if c.cell_type == "code" and "—" in c.source:
        c.source = c.source.replace("—", "-")
for c in C:
    if "—" in c.source:
        c.source = c.source.replace(" — ", ", ").replace("—", ", ")
assert not any("—" in c.source for c in C), "em dash survived"
nbf.write(nb, P)
print(f"de-AI'd 03 ({len(C)} cells); em dashes remaining:",
      sum(c.source.count("—") for c in C))
