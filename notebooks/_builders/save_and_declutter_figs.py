"""For every plotting cell in 01/02/03: shorten titles/labels (less clutter) and
insert savefig(dpi=600, bbox_inches='tight') into notebooks/figures/. Run from repo root."""
import re
import nbformat as nbf

# per notebook: {cell_idx: (savename, [(old,new),...], [(regex,repl,flags),...])}
EDITS = {
"01_regression_sanity.ipynb": {
  6:  ("01_regression_data",
       [('ax0.set_title("Regression: interleaved (x, y) token stream example")','ax0.set_title("Regression data")')], []),
  11: ("01_lambda_sweep",
       [('plt.xlabel("ridge regularizer (log scale)")','plt.xlabel("Λ (log)")'),
        ('plt.ylabel("held-out query MSE")','plt.ylabel("MSE")'),
        ('plt.title("Sweeping regularization strength")','plt.title("Ridge sweep (clean data)")')], []),
  14: ("01_cg_sweep",
       [('plt.xlabel("in-context examples (#)")','plt.xlabel("in-context examples")'),
        ('plt.ylabel("held-out query MSE  (log scale)")','plt.ylabel("MSE (log)")'),
        ('plt.title("MesaNet CG-step sweep")','plt.title("CG sweep")')], []),
},
"02_noisy_drifting_regression.ipynb": {
  5:  ("02_noise_data", [('ax.set_title("Perturbing the targets")','ax.set_title("Noise on targets")')], []),
  6:  ("02_drift_data",
       [('axL.set_title("Relative drift on token position")','axL.set_title("Drift: W rotates")'),
        ('axR.set_title("W components wandering (drift=0.1, one sequence)")','axR.set_title("W components (drift=0.1)")')], []),
  10: ("02_noise_sweep", [],
       [(r'title=f?"[^"]*"','title="Noise sweep"',0),(r'xlabel="[^"]*"','xlabel="noise σ"',0)]),
  13: ("02_cg_vs_noise", [], [(r'title=f?"[^"]*"','title="CG value vs noise"',0)]),
  17: ("02_drift_sweep", [],
       [(r'title=f?"[^"]*"','title=f"Drift sweep (noise={NOISE_FOR_DRIFT})"',0),(r'xlabel="[^"]*"','xlabel="drift"',0)]),
  20: ("02_cg_vs_drift", [], [(r'title=f?"[^"]*"','title="CG value vs drift"',0)]),
  26: ("02_forget_drift01", [],
       [(r'title=f?"[^"]*"','title=f"Forget sweep (drift={DRIFT_FGT})"',0),(r'xlabel="[^"]*"','xlabel="forget rate (1−δ)"',0)]),
  31: ("02_forget_drift02", [],
       [(r'title=f?"[^"]*"','title=f"Forget sweep (drift={DRIFT_F2})"',0),(r'xlabel="[^"]*"','xlabel="forget rate (1−δ)"',0)]),
  35: ("02_lambda_noise",
       [('plt.xlabel("ridge regularizer  Λ   (log scale)")','plt.xlabel("Λ (log)")'),
        ('plt.ylabel(f"held-out query MSE   (noise={NOISE_L}, drift=0)")','plt.ylabel("MSE")'),
        ('plt.title("Does regularization rescue MesaNet under heavy noise?")','plt.title("Ridge under noise")')], []),
  39: ("02_lambda_x_n",
       [('axL.set_xlabel("ridge regularizer  Λ  (log)")','axL.set_xlabel("Λ (log)")'),
        ('axL.set_ylabel(f"held-out MSE  (noise={NOISE_LN})")','axL.set_ylabel("MSE")'),
        ('axL.set_title("MSE vs Λ, one curve per n   (★ = best Λ)")','axL.set_title("MSE vs Λ per n")'),
        ('axR.set_xlabel("in-context examples  n")','axR.set_xlabel("n")'),
        ('axR.set_ylabel("regularization gain  (MSE@Λ_min − min_Λ MSE)")','axR.set_ylabel("ridge gain")'),
        ('axR.set_title("Does ridge pay off as n → d?")','axR.set_title("Ridge gain vs n")')], []),
},
"03_mqar.ipynb": {
  5:  ("03_mqar_data", [], [(r'ax\.set_title\("MQAR: writes[^\n]*\)','ax.set_title("MQAR sample")',0)]),
  6:  ("03_capacity_pressure", [], [(r'ax\.set_title\(f"MQAR under capacity.*?arrive"\)','ax.set_title("Capacity pressure")',re.DOTALL)]),
  7:  ("03_distractor_data", [], [(r'ax\.set_title\("MQAR with distractors[^\n]*\)','ax.set_title("With distractors")',0)]),
  16: ("03A_capacity", [], [(r'title=f?"[^"]*"','title=f"Capacity (gap={GAP_FIXED})"',0),(r'xlabel="[^"]*"','xlabel="n_pairs"',0)]),
  20: ("03A_capacity_control", [], [(r'title=f?"[^"]*"','title="Capacity: forgetting on vs off"',0),(r'xlabel="[^"]*"','xlabel="n_pairs"',0)]),
  24: ("03A_gap", [], [(r'title=f?"[^"]*"','title=f"Gap sweep (n_pairs={NPAIRS_GAP})"',0),(r'xlabel="[^"]*"','xlabel="gap"',0)]),
  28: ("03A_cg_headline", [],
       [(r'title=f?"[^"]*"','title=f"CG sweep (n_pairs={NPAIRS_HEAD})"',0),(r'xlabel="[^"]*"','xlabel="CG steps"',0),
        (r'\n *# annotate the analytic FLOPs.*?lw=0\.8\)\)','',re.DOTALL)]),
  32: ("03A_depth", [], [(r'title=f?"[^"]*"','title="Depth probe"',0),(r'xlabel="[^"]*"','xlabel="query index"',0)]),
  42: ("03B_distractors", [], [(r'title=f?"[^"]*"','title=f"Distractors (n_pairs={NPAIRS})"',0),(r'xlabel="[^"]*"','xlabel="n_distractors"',0)]),
  46: ("03B_forget", [], [(r'title=f?"[^"]*"','title=f"Forget gate (distractors={ND_FIXED})"',0),(r'xlabel="[^"]*"','xlabel="forget rate (1−δ)"',0)]),
  55: ("03C_heatmap", [], [(r'title=f?"[^"]*"','title="Mesa accuracy: δ × CG"',0)]),
  57: ("03C_delta_curves", [], [(r'title=f?"[^"]*"','title="δ curves (one per CG)"',0),(r'xlabel="[^"]*"','xlabel="forget rate (1−δ)"',0)]),
  58: ("03C_cg_curves", [], [(r'title=f?"[^"]*"','title="CG curves (one per δ)"',0),(r'xlabel="[^"]*"','xlabel="CG steps"',0)]),
},
}

for fname, cells in EDITS.items():
    nb = nbf.read("notebooks/"+fname, as_version=4)
    for idx,(name,str_reps,rx_reps) in cells.items():
        c = nb.cells[idx]; s = c.source
        assert c.cell_type=="code" and "plt.show()" in s, f"{fname} cell {idx} not a plot cell"
        for a,b in str_reps:
            assert a in s, f"{fname} cell {idx}: missing {a!r}"
            s = s.replace(a,b)
        for pat,repl,flags in rx_reps:
            s2 = re.sub(pat,repl,s,count=1,flags=flags)
            assert s2!=s, f"{fname} cell {idx}: regex no-op {pat!r}"
            s = s2
        s = s.replace('ylabel="answer-token exact-match accuracy"','ylabel="accuracy"')
        # drop any old explicit regression_* savefig
        s = "\n".join(ln for ln in s.split("\n") if 'savefig("figures/regression_' not in ln)
        # insert dpi-600 savefig right before show
        assert s.count("plt.show()")==1
        s = s.replace("plt.show()", f'plt.savefig("figures/{name}.png", dpi=600, bbox_inches="tight"); plt.show()')
        c.source = s
        compile(s.replace("%matplotlib inline","pass"), f"<{idx}>", "exec")
    nbf.write(nb, "notebooks/"+fname)
    print(f"{fname}: edited {len(cells)} figure cells")
print("done")
