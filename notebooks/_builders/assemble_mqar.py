"""Restructure the notebook suite (no execution):
  - combine 03 (capacity) + 04 (distractors) + 05 (δ×CG) -> 03_mqar.ipynb
    with MQAR dataset-inspection cells inline at the top (moved from 00).
  - move regression inspection -> 01; noise/drift inspection -> 02.
  - delete 00, the old 03_mqar_capacity, 04, 05.
All reused cells are stripped of outputs; per-section vars are namespaced so the
combined notebook runs cleanly top-to-bottom.
"""
import re
import os
import nbformat as nbf

NB = "notebooks/"

def load(p):
    return nbf.read(NB + p, as_version=4)

def md(s):
    return nbf.v4.new_markdown_cell(s)

def code(s):
    return nbf.v4.new_code_cell(s)

def clone(cell):
    """Output-stripped copy of a cell."""
    if cell.cell_type == "markdown":
        return nbf.v4.new_markdown_cell(cell.source)
    return nbf.v4.new_code_cell(cell.source)

def rename_code(cell, subs):
    """Apply (pattern, repl) regex subs to a CODE cell only; return a fresh cell."""
    if cell.cell_type != "code":
        return clone(cell)
    s = cell.source
    for pat, repl in subs:
        s = re.sub(pat, repl, s)
    return nbf.v4.new_code_cell(s)

def demote_title(cell, new_first_line):
    """Replace the first line of a markdown cell (the H1 title) with a section header."""
    lines = cell.source.split("\n")
    lines[0] = new_first_line
    return nbf.v4.new_markdown_cell("\n".join(lines))

def is_show_def(cell):
    return cell.cell_type == "code" and cell.source.lstrip().startswith("def show(")

# --------------------------------------------------------------------------- #
# extract dataset-inspection cells from 00 (by id)
# --------------------------------------------------------------------------- #
nb00 = load("00_datasets.ipynb")
c00 = {c.get("id"): c for c in nb00.cells}

def strip_savefig(cell):
    s = "\n".join(l for l in cell.source.split("\n") if "savefig" not in l)
    return nbf.v4.new_code_cell(s)

# ===========================================================================
# 1) COMBINED MQAR NOTEBOOK  ->  notebooks/03_mqar.ipynb
# ===========================================================================
cells = []
cells.append(md(
    "# 03 · MQAR — recall, capacity, interference, and the δ×CG surface\n\n"
    "Multi-query associative recall (MQAR) is the memory task where MesaNet's exact "
    "full-history solve is *supposed* to pay off. This notebook runs three experiments on it:\n\n"
    "- **A · Capacity** — recall vs memory load `n_pairs`; plus a control disentangling "
    "state-capacity from forgetting.\n"
    "- **B · Distractors** — recall under never-queried competing bindings, and the forget gate "
    "(δ) under interference.\n"
    "- **C · δ × CG interaction** — the joint surface of the two Mesa levers (retention × solve depth).\n\n"
    "Each section trains its **own** models (different training distributions) and is self-contained; "
    "the dataset is inspected inline just below. (This notebook merges the former 03/04/05.)"
))
# shared imports
cells.append(code(
    "%matplotlib inline\n"
    "import sys, os, math\n"
    'sys.path.insert(0, os.path.abspath(".."))  # project root: makes `compare` importable\n\n'
    "import numpy as np\n"
    "import torch\n"
    "import matplotlib.pyplot as plt\n"
    "from matplotlib.colors import ListedColormap, BoundaryNorm\n"
    "from matplotlib.patches import Patch\n\n"
    "from synthtasks.mqar import make_mqar\n"
    "from compare.experiments import (\n"
    "    MQARConfig, train_mqar, train_mqar_models, mqar_sweep_rows, mqar_cg_rows,\n"
    "    mqar_position_rows, mqar_flops_note, plot_mse_vs, plot_capacity_forget_control,\n"
    "    mqar_delta_cg_sweep, plot_delta_cg_heatmap,\n"
    ")\n\n"
    "np.set_printoptions(precision=3, suppress=True)\n"
    "SEED = 0\n"
    'DEVICE = "cuda" if torch.cuda.is_available() else "cpu"'
))
# shared show() helper (one copy for all sections)
cells.append(code(
    'def show(rows, x, ykey="acc"):\n'
    '    """Compact accuracy table: one row per model label, columns = the swept axis `x`."""\n'
    "    labels = []\n"
    "    for r in rows:\n"
    '        if r["label"] not in labels:\n'
    '            labels.append(r["label"])\n'
    "    xs = sorted({r[x] for r in rows if r.get(x) is not None})\n"
    '    print(f"{x:>16}  " + "  ".join(f"{v:>6}" for v in xs))\n'
    "    for lb in labels:\n"
    '        dd = {r[x]: r[ykey] for r in rows if r["label"] == lb and r.get(x) is not None}\n'
    '        body = "  ".join((f"{dd[v]:6.3f}" if v in dd else "    --") for v in xs)\n'
    '        print(f"{lb:>16}  {body}")'
))
# dataset inspection (moved from 00)
cells.append(md(
    "## MQAR data — what the model sees\n\n"
    "Before the experiments, three inspection views of the generator (no models): a small "
    "write→query→answer sample, the same under capacity pressure (large `n_pairs`), and with "
    "distractors (competing never-queried bindings)."
))
for cid in ("b55dfa77", "aa3f2ee3",        # MQAR sample (print + role viz)
            "42f8b188",                      # capacity pressure
            "6522e819"):                     # distractors
    cells.append(clone(c00[cid]))

# --- Section A: capacity (from 03_mqar_capacity.ipynb) ---
nb03 = load("03_mqar_capacity.ipynb")
A_SUBS = [
    (r"\bmqcfg_off\b", "cfg_cap_off"), (r"\bmqcfg\b", "cfg_cap"),
    (r"\bmodels_off\b", "models_cap_off"), (r"\bmodels\b", "models_cap"),
    (r"\bCMP\b", "cmp_cap"),
]
for i, c in enumerate(nb03.cells):
    if is_show_def(c):
        continue                              # use the shared show()
    if i == 0:
        cells.append(demote_title(c, "## A · Capacity — where exact full-history fit should pay off"))
        continue
    cells.append(rename_code(c, A_SUBS))

# --- Section B: distractors (from 04_mqar_distractors.ipynb) ---
nb04 = load("04_mqar_distractors.ipynb")
B_SUBS = [(r"\bmqcfg\b", "cfg_dist"), (r"\bmodels\b", "models_dist")]
for i, c in enumerate(nb04.cells):
    if is_show_def(c):
        continue
    if i == 0:
        cells.append(demote_title(c, "## B · Distractors — interference & the forget gate"))
        continue
    cells.append(rename_code(c, B_SUBS))

# --- Section C: δ × CG (from 05_delta_cg_interaction.ipynb) ---
nb05 = load("05_delta_cg_interaction.ipynb")
C_SUBS = [(r"\bmqcfg\b", "cfg_dxcg"),
          (r"(PER_CELL_STEPS\s*=\s*)2000", r"\g<1>3000")]   # user-chosen budget
for i, c in enumerate(nb05.cells):
    if is_show_def(c):
        continue
    if i == 0:
        cells.append(demote_title(c, "## C · δ × CG interaction — do Mesa's two levers compose?"))
        continue
    cells.append(rename_code(c, C_SUBS))

out = nbf.v4.new_notebook()
out.cells = cells
out.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}}
nbf.write(out, NB + "03_mqar.ipynb")
print(f"wrote 03_mqar.ipynb with {len(cells)} cells")

# ===========================================================================
# 2) regression inspection -> 01 ; noise/drift inspection -> 02
# ===========================================================================
def insert_after(nb, after_idx, new_cells):
    nb.cells[after_idx + 1:after_idx + 1] = new_cells
    return nb

nb01 = load("01_regression_sanity.ipynb")
reg_md = md("## The regression task — what the model sees\n\n"
            "Inline inspection of the generator (no models): the interleaved (x, y) token stream and "
            "the scalar target at each scored query position.")
reg_setup = code("from synthtasks.regression import make_regression  # inline dataset inspection")
insert_after(nb01, 2, [reg_md, reg_setup, clone(c00["3cf914a9"]), strip_savefig(c00["fe53e4e0"])])
nbf.write(nb01, NB + "01_regression_sanity.ipynb")
print(f"updated 01 -> {len(nb01.cells)} cells")

nb02 = load("02_noisy_drifting_regression.ipynb")
nd_md = md("## The difficulty knobs — what noise & drift do to the data\n\n"
           "Inline inspection (no models): label noise perturbs the targets off the clean line; "
           "drift rotates the ground-truth map `W` across the sequence.")
nd_setup = code("from synthtasks.regression import make_regression  # inline dataset inspection")
insert_after(nb02, 2, [nd_md, nd_setup, clone(c00["3eb03924"]), clone(c00["784dfc9c"])])
nbf.write(nb02, NB + "02_noisy_drifting_regression.ipynb")
print(f"updated 02 -> {len(nb02.cells)} cells")

# ===========================================================================
# 3) delete the superseded notebooks
# ===========================================================================
for f in ("00_datasets.ipynb", "03_mqar_capacity.ipynb",
          "04_mqar_distractors.ipynb", "05_delta_cg_interaction.ipynb"):
    p = NB + f
    if os.path.exists(p):
        os.remove(p)
        print(f"deleted {f}")
print("done")
