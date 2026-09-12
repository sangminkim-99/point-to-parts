# Agent status board

Shared state for the human and Claude (Claude Code). Codex (GPT) left the
project on 2026-09-11 when its token budget ran out; Claude is now the single
agent (lead, integrator and worker).
Read this file **and** `doc/agent_dialogue.md` before starting any task.
Update the "Status" and "Next" rows when you finish a unit of work.

Last updated: 2026-09-11 21:30 EDT by Claude.
Base commit: `bec3faf`.

## Objective

Make RGB-D part discovery/tracking reliable enough to supply persistent
part-local point clouds and credible joint estimates for mesh-free
manipulation. Current checkpoint, confirmed progress, failed hypotheses and
priority order: `doc/demo_checkpoint_20260911.md`.

## Ownership

| Area | Owner | Files |
| --- | --- | --- |
| Everything | Claude | all |
| This board and the dialogue log | append-only record | `doc/agent_status.md`, `doc/agent_dialogue.md` |

## Status

| Item | State |
| --- | --- |
| Demo checkpoint (frozen split gate + live fallback, Viser diagnostics, URDF viewer) | committed `c3db1c2`, pushed |
| Demo worker outputs (thin-slab turnover control, growth/appearance probes, URDF audit) | integrated `9bd148b` |
| Lifecycle ledgers + `merge_probation` (default off) + diagnostics dumps + matched runner | integrated `bec3faf`; runner reproduces `doc/newborn_part_probation.md` |
| Split-time consistency guard (`split_reprojection_consistency`) | **rejected** by the 09-10 15:44 review; not integrated |
| Validation worktrees under `.claude/worktrees/` (14) | removed 2026-09-11; each one's uncommitted state is a "Snapshot uncommitted worktree state" commit on its own branch (`worktree-articulation-demo`, `wt-articulation-probation`, `frozen-*`, `gate-min-integration`, `growth-ablation`, `jm-rerun`, `kf-baseline-val`, `orphan-snapshot`, `relock-guard-val`, `turnover-diag`, `ikea-idrerun`) |
| `test/object` | 175 passing |

## Next

Order set by `doc/demo_checkpoint_20260911.md`:

1. ikeasmall02 parent graph: 5 parts vs 3 GT, two joint rows on rb2, both
   parents wrong. Correct the kinematic graph before adding any more gates.
   Candidate route: pairwise relative-motion joint discovery (user-approved,
   see the 09-10 relative-motion dialogue entry).
2. URDF export defect 2 from `doc/urdf_export_audit.md`: `model.npz` poses
   and the URDF chain disagree by the joint-fit residual (up to ~5 deg /
   40 mm) with no test that can catch it. Add the test and disclose the bound
   in `metadata.json`.
3. Thin-slab turnover: multiple plausible poses + uncertainty reporting, per
   `doc/thin_part_turnover.md`. Appearance is a veto signal only
   (`doc/thin_turnover_control.md`).
4. Gaussian duplication under rigid translation (`doc/gaussian_growth_probe.md`):
   trace `dense_model.grow/occupancy`; growth can be suspended during pure
   translation at zero pose cost.

## Lifecycle instrumentation (in `bec3faf`)

No behaviour change; logging and ledgers only.

- `NaivePart.birth` — immutable birth frame, distinct from `born`, which
  `_merge_rigid` resets as a settle clock.
- `NaivePart.obs_prov` — per-observation provenance (frame, parent, live or
  rebuild), parallel to `joint.A`; this is what `merge_probation` scores on.
- `birth_log`, `death_log`, `merge_log` on the tracker — every part_id that
  came into being, every one merged away, and every merge decision including
  the ones that kept a part or were withheld (`on_probation`).
- `alive_frames` / `seen_frames` per part_id — existence and measurement
  counted separately.
- `lifecycle_summary()` and `replay.py --dump-lifecycle <path>.json` — the
  GT-free machine-readable survival record.
- `replay.py --dump-surface`, `--dump-split-reproj`, `--reference-part` —
  opt-in diagnostics behind the drawer-gate and surface-ownership reports.

## Running anything in this repo

`gsplat` JIT-compiles its CUDA extension on import and needs `CUDA_HOME`,
`ninja` on `PATH`, and (for a rebuild) `CPATH` and `TORCH_CUDA_ARCH_LIST`.
Without them gsplat disables itself and the run dies much later, inside the
dense path, as `AttributeError: 'NoneType' object has no attribute
'CameraModelType'`. What works, with the repo-local extension cache that
already holds a built `gsplat_cuda.so`:

```sh
P=$HOME/miniconda3/envs/point2pose_model
env CUDA_HOME=$P PATH=$P/bin:$PATH CPATH=$P/targets/x86_64-linux/include:$P/include \
    TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2 TORCH_EXTENSIONS_DIR=$PWD/results/torch_extensions \
    $P/bin/python -u -m examples.multi_part.replay --seq-dir ... --method naive ...
```

Two traps hit on 2026-09-11: in zsh `env $VARS python` does not word-split
`$VARS`, so only a garbled `CUDA_HOME` is set and the same error appears; and
`replay.py --stride` defaults to **2**, while every table in
`doc/newborn_part_probation.md` is stride 1 (the runner sets it). At stride 2
laptop_hinge never splits under `reprojection_split.yaml`.

Tests need the same environment; `test/object/test_sim_sequence.py`
imports `sapien` at module scope and errors on collection in any other env.

## Known constraints

- `/home/smkim/data/mp/` holds only take01 and lift01, both with a single
  binary union mask and **no ground truth**. ID-survival claims that need
  truth lean on `results/partnet_dev_v1/`; real per-part GT is RBO
  (`doc/rbo_sam2_demo.md`, SAM2 masks with a GT-derived init box).
- `scripts/sim/ablate_reprojection.py` requires `<case>/meta.json` and a GT
  `[eval]` line; `examples/multi_part/run_probation_experiment.py` is the
  GT-free matched runner.

## Conventions

- Large outputs under `results/<topic>_v<N>/` (gitignored).
- New tracker options default to off and live as fields on `NaiveConfig`
  (`examples/multi_part/naive.py`), settable via `--set key=value`.
- Reprojection gating stays **on** for probation comparisons
  (`--config reprojection_split.yaml`), per the earlier ablation.
- Report meaningful progress to the user in Korean; keep matched ablations
  and negative results in the dialogue log.
