# linear_rnn

A small set of experiments comparing two linear-RNN sequence layers: MesaNet and
Gated DeltaNet.

Both layers process a sequence by keeping a fixed-size memory and updating it one
token at a time. They differ in how they read that memory back. Gated DeltaNet
does one cheap update per step. MesaNet instead solves a small least-squares
problem at each step, which costs more but uses the whole history at once. The
amount of work MesaNet does is adjustable, so you can dial it from "cheap, like
Gated DeltaNet" up to "full exact solve".

The question this repo looks at: when is the extra work worth it? On some tasks
the exact solve clearly helps; on others it does not, and the cheaper layer is
just as good. The goal is to find where that line is, and why, rather than to
declare one layer better overall.

To keep the comparison fair, the two layers are dropped into the same small model
and trained the same way. The only thing that changes between runs is the mixing
layer itself. The tasks are simple, made-up problems where the right answer is
known, each one built to stress a single thing: recalling stored facts, fitting a
noisy line, tracking a moving target, and so on.

## Layout

- `synthtasks/` — the toy tasks and how they are scored.
- `compare/` — the model, the training loop, and the experiment runners.
- `notebooks/` — the experiments themselves, written up step by step.

## Running it

The two real layers need a GPU. The data, the scoring, and a stand-in CPU layer
run anywhere. Set up the environment with `uv sync`, then open the notebooks in
order.

## What we found

The extra work pays off when there is enough clean data and a lot to remember.
MesaNet reaches low error from fewer examples, and on the recall task it holds up
better than Gated DeltaNet as the number of stored facts grows. Most of that gain
comes from just a couple of solve steps; doing more barely helps. The advantage
shrinks or disappears when the target keeps moving or when data is scarce and
noisy, and there the cheaper layer is about as good. In those cases the thing that
actually matters is not how hard MesaNet solves but how much it forgets, or simply
how much data it has. A quantity that MesaNet computes for free turns out to flag
whether a recall was right.

These are small models run once each, so read them as a direction, not a verdict.
