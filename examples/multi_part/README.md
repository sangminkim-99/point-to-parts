# Live multi-part discovery

RGB-D in, parts out. The object starts as one rigid body with a single box; as it
articulates the parts separate on their own, and each one gets a dense model, an
oriented box, a 6-DoF pose and a joint. No CAD model, no part count, no category
prior. Everything is causal — nothing looks at a future frame.

```
examples/multi_part/
  streaming.py             the method as an online state machine
  replay.py                run it over a recorded sequence (no camera needed)
  realsense_multipart.py   run it live from a RealSense
```

## Run it

Live:

```bash
python -m examples.multi_part.realsense_multipart --n-points 224
# left click the object, 's' to start, 'r' to reset, 'q' to quit
```

On a recorded sequence, which is also how the numbers below were measured:

```bash
python -m examples.multi_part.replay \
  --seq-dir /path/to/sequence --n-points 224 \
  --out debug/multi-parts/runs/stream/out.mp4
```

## It is not real time — measured

On the simulated pliers sequence, one RTX-class GPU, 3683 gaussians:

| query points | TAPIR resolution | fps | outcome |
|---|---|---|---|
| 400 | 480 | 2.9 | 2 parts |
| 400 | 256 | 3.9 | 2 parts |
| **224** | **480** | **3.8** | **2 parts, revolute joint — the recommended setting** |
| 224 | 256 | 4.5 | 3 parts (over-segmented, one of 36 gaussians) |
| 128 | 480 | 7.1 | **never splits at all** |
| 128 | 256 | 6.1 | 2 parts |

Where the time goes at 400 points, per frame:

| stage | ms | share |
|---|---|---|
| point tracking (TAPIR) | 220 | 63% |
| hypothesis RANSAC | 73 | 21% |
| dense assignment + per-part fits | 64 | 18% |

**Point tracking dominates, and it cannot simply be turned down.** TAPIR costs
about 0.55 ms per query point, so the obvious lever is to use fewer — but 128
points at full resolution never separates the parts at all. Keypoint density is
the binding constraint on part *discovery*, which was measured long before this
demo existed: at 30 points the sparse detector found nothing, at 250 it found five
parts. Buying frame rate with query points buys it out of the thing the demo is
for.

The dense side is genuinely cheap: about 22 ms/frame and nearly independent of
cloud size, because it is dominated by the kernel launches of the ICP loop rather
than by the number of gaussians. Getting to interactive rates means making point
tracking cheaper, not making the assignment cheaper.

Skipping tracker frames (`--hyp-every N`) is available but defaults to 1, because
the sparse tracks are **not** only used before the split: after it, each part fits
its own RANSAC hypothesis from the tracks that sit on it, which is what a thin,
fast-rotating part depends on — offline, removing it is where the scissors blade
and the eyeglasses temple diverged.

## TAPIR refinement iterations

`num_pips_iter` defaults to 1. It is the single biggest speed lever and it is not
free:

| | pips=4 | pips=2 | pips=1 (default) |
|---|---|---|---|
| pliers | 3.5 fps, energy 0.0027/0.0017, both revolute | 4.4 fps, 0.0031/0.0019 | 5.8 fps, 0.026/0.019, one joint and it comes out prismatic |
| storage | 4.1 fps, 0.037/0.005 | 5.5 fps, 0.038/0.004 | 8.2 fps, 0.046/0.038, one part shrinks to 537 gaussians |
| eyeglasses | 3.1 fps, **3 parts** | 4.5 fps, 2 parts | 5.4 fps, 2 parts |

Tracking energy is roughly 10x worse at 1 than at 4, and eyeglasses already loses
a part at 2. Raise it with `--pips-iter 4` when the result matters more than the
frame rate. The offline benchmark in `experiments/articulated/` leaves it at the
TAPIR default of 4.

## Environment

gsplat compiles its CUDA kernels on first use (about 80 s, then cached), which
needs the toolchain visible:

```bash
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include
export TORCH_CUDA_ARCH_LIST=8.9   # your GPU's compute capability
```

## Why part discovery used to appear so late

A split attempt is not per-frame work and has nothing to do with the number of
hypotheses. It is the grouping: log-odds labels, agglomerative clustering of the
co-association matrix, winner-set clustering, a rigid-merge variant of each, and
the model selection over all of them. Measured, one attempt:

| co-association sample | cost per attempt |
|---|---|
| 4000 | 660&ndash;860 ms |
| 2000 | 192 ms |
| 1000 | 46&ndash;52 ms |
| 500 | 10 ms |

At 4000 samples every attempt stalls the stream for most of a second, which is
why the split used to be gated to "not before frame 25, then once every 10". That
gate *was* the latency. Lowering the sample count to 1000 makes an attempt cheap
enough to gate at (12, 3) instead, and on pliers that pulled the split from frame
31 to 25 while also improving the result: tracking energy went from 0.02&ndash;0.04
to 0.0027/0.0017, because splitting earlier leaves more frames for each part's
model to grow.

Checked at 1000 against 4000 on pliers, storage and eyeglasses: same part count,
same joint types, comparable energies. The one real difference is storage, where
co-association wins the selection — its initial grouping covers 5% of the cloud
instead of 19%, since only sampled gaussians can be labelled, and the final parts
match only because growth fills the rest back in.

The per-frame assignment, by contrast, was never the problem. It is not
rasterisation: the rendering path exists but is unused, because rendering the
whole cloud under every hypothesis conflates them (on RBO one hypothesis then
"explained" 97.8% of pixels). What runs is a point-wise projection of each
gaussian under each hypothesis against the observed depth, and it costs 22 ms.

## How the state machine works

**RIGID.** One body, one box. Every frame proposes rigid-motion hypotheses from a
trailing window of the sparse tracks, assigns each gaussian a soft posterior over
them, and refines each hypothesis against the surface it should explain. Evidence
accumulates three ways at once: per-gaussian log-odds, pairwise co-association,
and the decisive winner sets.

Every `regroup_every` frames the accumulated evidence is offered to the same
ground-truth-free model selection the offline benchmark uses — four candidate
groupings, scored by how much better splitting explains the recent posteriors than
not splitting, with coverage breaking ties and a rigid-merge variant of each. A
split is committed only when it beats treating the object as one body.

**SPLIT.** Each part is tracked forward only. Candidates are the previous pose,
the current hypotheses, fractional screw extrapolations of the last increment, a
RANSAC fit on the part's own sparse tracks, and a sweep of the fitted joint; each
is refined by projective ICP against the object mask minus what the other parts
occupy, and the one that best explains the surface wins, subject to a per-frame
step cap. Each part's dense model grows as unexplained surface appears, gated on
that part's own tracking energy, and its joint is fitted and refitted online.

## Known gaps

- **Frame rate**, as above.
- The split threshold is the offline one. A live scene where the object never
  articulates simply stays in RIGID, which is correct, but there is no feedback
  telling the user "move it more".
- The joint is fitted per part against one parent; a chain deeper than two links
  is expressed relative to the largest other part rather than its true parent.
