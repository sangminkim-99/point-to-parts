# Gaussian duplication under rigid whole-object motion — measured

Owner: Claude (demo worker). Files: `scripts/sim/gaussian_growth_probe.py`,
the `--translation-axis` extension of `scripts/sim/thin_turnover_control.py`,
this report. **No tracker files edited**; growth is disabled through the
existing config gate (`grow_every=0`, `naive.py:576`). Tracker snapshot:
root checkpoint state, naive.py sha `48a2d015…` (root has advanced since with
uncommitted tracking-worker edits, not included).

## Question

User report: the dense model **duplicates geometry during vertical
whole-object motion**. Competing explanations:
(a) **duplication** — the posed model fails to absorb the motion, so
already-modelled surface is re-added; (b) **real coverage** — the motion
genuinely reveals new surface.

## Method

GT stays **eval-only**; the tracker gets rgb, depth and the **oracle binary
union mask** (`seg != 255`) — no per-part input. The legitimacy ceiling for
growth is computed from GT alone: 5 mm object-frame surface voxels (masked
depth through the GT pose) never visible before. Controls (all rigid slab,
fixed camera, 90 frames, noise 0, `results/thin_turnover_v1/`):

- **`vertical`** — pure 8 cm vertical lift, **0° rotation**. Same faces stay
  visible the whole time, so GT new-view coverage is ~0 *by construction*.
- **`static`** — no motion at all: isolates warm-up growth.

Each probed twice with the unmodified tracker: `baseline` (`dense=true`) and
`growth_off` (`dense=true grow_every=0`). Probe:
`python -m scripts.sim.gaussian_growth_probe <render_dir> --out <dir>`;
JSONs under `results/thin_turnover_v1/{vertical,static}_probe/`.

## Results

| | GT new-view ceiling (after f0) | model growth (baseline) | growth by thirds | pose rot/trans err (median) |
| --- | --- | --- | --- | --- |
| static | **0** voxels | +179 (3.6%) | 102 / 73 / **4** — converges | 0.06° / 0.1 mm |
| vertical | **84** voxels | **+1024 (20.6%)** | 602 / 239 / **183** — persists | 0.61° / 7.8 mm |

- `growth_off`: zero growth (gate works), and **pose error is bit-identical to
  baseline** in both controls.
- Unassigned count stays **0** throughout: carve never removes any of the
  added Gaussians — duplicates are permanent.
- Growth fires on the `grow_every=4` cadence with pose error at growth frames
  ≈ the overall median (8.4 vs 7.8 mm) — it is not triggered by tracking
  failures or error spikes.

## Verdict

**Duplication, not real coverage — and not pose error absorbing the motion.**

1. Under vertical motion the model adds **1024** Gaussians against a GT
   ceiling of **84** legitimately-new voxels: ≥ 92% of the growth duplicates
   already-modelled surface. Static shows the same mechanism only as a
   converging ~179-Gaussian warm-up; **motion sustains it indefinitely**
   (+183 in the final third, where GT new coverage is ~0).
2. The pose is **not** being absorbed into the model: tracking is good
   (0.61° / 7.8 mm over an 80 mm lift) and *identical* with growth off — so
   the dense growth feeds nothing back into pose here; it only accumulates.
3. Mechanism (**hypothesis**, consistent with all three observations but not
   traced into the code): with ~8 mm steady pose error (≈ 1.6 voxels), the
   occupancy test that gates `grow_parts` keeps finding "uncovered" pixels at
   the model's misaligned silhouette each motion frame and re-adds them; the
   3 cm carve margin then protects the duplicates forever. Confirming that
   requires stepping through `dense_model.grow/occupancy`, which is the
   tracking owner's file.

**Cost of the mitigation measured here:** on these rigid controls,
`grow_every=0` sacrifices nothing — pose identical, and no new surface existed
to miss. That is NOT a recommendation to disable growth generally (a turnover
or drawer-opening genuinely reveals surface); it bounds the harm: growth can
be suspended during pure translation phases at zero pose cost.

## Frozen split geometry (`619ed71`) validated on the rigid controls

Worktree aligned to exactly `619ed71`; runs compare
`reprojection_split.yaml` vs `reprojection_split_frozen.yaml`
(`split_reprojection_frozen: true`), both `dense=true`, isolated build cache.
JSONs: `results/thin_turnover_v1/{rigid,vertical}_frozen_619ed71/growth_probe.json`.

| control | config | false splits | rot err obs/held (median) | growth | step ms (med/p90) |
| --- | --- | --- | --- | --- | --- |
| rigid turnover | baseline | 0 (1 part) | 179.95° / 45.22° | +1014 (20.4%) | 121 / 153 |
| rigid turnover | **frozen** | 0 (1 part) | 179.95° / 45.22° | +994 (20.0%) | 107 / 143 |
| vertical lift | baseline | 0 (1 part) | 0.61° / — | +1024 (20.6%) | 110 / 131 |
| vertical lift | **frozen** | 0 (1 part) | 0.61° / — | +1059 (21.3%) | 109 / 127 |

- **No regression**: false splits stay 0, pose errors are identical to the
  digit (obs/held-conditioned alike), observed coverage unchanged (0.86 /
  1.00), and no runtime penalty is evident (single runs; machine-load spread
  on this box is ~30%, so no timing claim beyond "not slower").
- **But the frozen path never fired**: a full-log run shows **0**
  `geometry=frozen@` lines on the rigid turnover — no split reaches the
  evidence stage on these controls, so retained snapshots are never consulted.
  This validates the variant as *safe to enable* here, **not** that it helps;
  its benefit case (drawer splits surviving map maintenance, 58-vs-1 in
  `doc/frozen_split_evidence.md`) needs an articulated sequence, which is the
  tracking owner's evaluation.
- Growth differs by ±2–3% between baseline and frozen (−20 / +35 Gaussians)
  even though back-to-back baseline runs reproduce exactly; a small
  interaction of the snapshot bookkeeping with growth is plausible but
  **unadjudicated** — flagged, not explained.
- The turnover/vertical duplication findings above reproduce at `619ed71`
  (rigid +1014, vertical +1024 baseline), so `9ce1cde`'s grown-scale change
  did not alter the duplication picture either.

## Gate stress controls: forcing proposals under noise + occlusion

The rigid validation above never exercised the gate (0 proposals), and a safe
verdict must not be inferred from zero proposals. Stress controls added
(occluder + `--occluder-mode dwell` + depth-noise dial in
`thin_turnover_control.py`; oracle union masks throughout — the occluder is
not a part, so its pixels are honestly *excluded* from the mask while depth
shows the bar, like a hand crossing a real capture). Baseline vs frozen always
share the identical rendered input (same seed, same files).

| control (stress) | proposals | gate decisions | false splits | rot err obs (med/max) |
| --- | --- | --- | --- | --- |
| vertical + noise(1.5 mm@1 m) + sweep occluder (46% hidden) | 0 / 0 | — | 0 / 0 | 0.86°/2.97° both |
| turnover + noise + sweep occluder | 0 / 0 | — | 0 / 0 | 179.93°/180° both |
| vertical + noise(1 cm@1 m) + **dwell** occluder (object cut in two, 15 frames) | 0 / 0 | — | 0 / 0 | 0.82°/5.96° both |
| **moving_laptop** (rigid yaw+lift, richer geometry) | **1 / 1** | reject / reject, `gain_below_threshold` | 0 / 0 | 0.21°/0.66° both |

- **The frozen path finally fired and was measured head-to-head**: on
  moving_laptop's f50 proposal the baseline scores `geometry=live`
  (gain 0.001, support [0.94, 0.32]) and the frozen variant scores
  `geometry=frozen@6` — a retained frame-6 snapshot actually consulted —
  (gain 0.002, support [0.94, 0.30]). **Same decision, reject**, which is
  correct on this rigid GT. Evidence values shift slightly (support −0.025);
  decision-identical here, but that shift is the thing to watch on borderline
  articulated proposals.
- **Rigid slabs never generate proposals, even under heavy stress.** Splitting
  is proposed upstream by coherent differential point motion, which occlusion
  and noise do not fabricate; a bar cutting the object in two for 15 frames
  plus 6× sensor noise still yields zero candidates. So the split *proposal*
  stage — not the reprojection gate — is what keeps rigid slabs at one part,
  and the gate can only be exercised where geometry is rich enough
  (moving_laptop) or motion is genuinely articulated (drawer/hinge — tracking
  owner's evaluation).
- **Frozen remains regression-free under stress**: identical pose errors and
  part counts on every pair, growth within ±2%, runtime not slower (single
  runs; no strong timing claim).
- **Side-finding (calibration, echoes the turnover baseline):** with realistic
  noise the turnover's *held* safety net vanishes — observed coverage rises
  from 0.86 (clean; 17 held frames at edge-on) to **1.00**, i.e. the tracker
  stops holding at all and claims observed straight through the flip at ~180°
  error. Noise makes the false-fresh problem worse, not better.

Artifacts: `results/thin_turnover_v1/{stress_vertical,stress_turnover,stress2_vertical,moving_laptop}_gate/growth_probe.json`
(per-frame rows + `gate_events` with parsed gain/accept/mode/geometry/reason).

## Limitations

- Two rigid sim controls, noise 0, oracle masks; no real-capture replication
  of the user's report (their capture has SAM2 masks + sensor noise, likely
  worse).
- The mechanism in §3 is untraced; per-Gaussian creation-time provenance would
  settle it but needs tracker-side instrumentation (not mine to add).
- `static` warm-up growth (+179) is not adjudicated good/bad here; it
  converges, so it is not the reported problem.

## Reproduce

```sh
P=$HOME/miniconda3/envs/point2pose_model
ENV="env CUDA_HOME=$P PATH=$P/bin:$PATH CPATH=$P/targets/x86_64-linux/include:$P/include \
     TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2 TORCH_EXTENSIONS_DIR=<worktree>/results/torch_extensions"
$ENV python -m scripts.sim.thin_turnover_control --out results/thin_turnover_v1/vertical \
    --variant rigid --frames 90 --turnover-deg 0 --translation 0.08 --translation-axis z
$ENV python -m scripts.sim.gaussian_growth_probe results/thin_turnover_v1/vertical \
    --out results/thin_turnover_v1/vertical_probe
```
