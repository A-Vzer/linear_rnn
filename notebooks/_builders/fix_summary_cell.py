"""Repair cell 35 of 03_mqar.ipynb: replace the mangled (commented-out) control
block with correct Python, and drop the inflated 'genuine benefit' line from the
baked output. Source and output are kept consistent."""
import re
import nbformat as nbf

P = "notebooks/03_mqar.ipynb"
nb = nbf.read(P, as_version=4)
cell = nb.cells[35]

# correct control block (raw string: real newlines between statements, literal \n in the f-string)
newblock = r'''# control: did forgetting hide part of the high-load drop? (shift when delta -> 1)
print(f"\n(control) forgetting off vs matched, accuracy at n_pairs=64 (gap={GAP_FIXED}):")
ko = {r["label"]: r["acc"] for r in cap_rows if r["n_pairs"] == 64}
oo = {r["label"]: r["acc"] for r in cap_rows_off if r["n_pairs"] == 64}
for lb in ko:
    d = oo[lb] - ko[lb]
    if abs(d) < 0.03:
        tag = "(decay-robust: real capacity)"
    elif d > 0:
        tag = "(off > matched: its drop was partly forgetting)"
    else:
        tag = "(decay-sensitive but seed-unstable: off < matched here)"
    print(f"    {lb:16} matched {ko[lb]:.2f} -> off {oo[lb]:.2f}  ({d:+.2f})  {tag}")

'''

src = cell.source
i1 = src.index("# control")
i2 = src.index("pos_first =")
cell.source = src[:i1] + newblock + src[i2:]

# the control block must now be real code, not a comment
assert "\nfor lb in ko:\n" in cell.source, "control block still mangled"
compile(cell.source, "<cell35>", "exec")

# drop the genuine-benefit line from the baked output (real newline now)
out = [o for o in cell["outputs"] if o.get("output_type") == "stream"][0]
out["text"] = re.sub(r"\n *-> genuine exact-solve benefit[^\n]*", "", out["text"])
assert "genuine exact-solve benefit" not in out["text"]
assert "—" not in cell.source

nbf.write(nb, P); nbf.validate(nb)
print("fixed cell 35. control section of output:")
for ln in out["text"].split("\n"):
    if "matched" in ln or "(control)" in ln or "genuine" in ln:
        print("  " + ln)
