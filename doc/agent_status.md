# Agent status board

Shared state for the human, Codex (lead) and Claude (Claude Code).
Read this file **and** `doc/agent_dialogue.md` before starting any task.
Update the "Status" and "Next" rows when you finish a unit of work.

Last updated: 2026-09-10 12:10 EDT by Claude.
Base commit: `679ab7a`.

## Objective

Make RGB-D part discovery/tracking reliable enough to supply persistent
part-local point clouds and credible joint estimates for mesh-free
manipulation. The active experiment is **newborn-part merge diagnosis and
bounded probation**, the follow-up named at the end of
`doc/realsense_recorded_replay.md`.

## Ownership

Do not edit a file another agent owns. If you need a change there, write the
request into `doc/agent_dialogue.md` and let the owner make it.

| Area | Owner | Files |
| --- | --- | --- |
| Newborn-part survival experiment | Claude | `examples/multi_part/naive.py`, `doc/newborn_part_probation.md`, experiment runner, focused tests |
| Integration and review | Codex | — (reviews Claude's edits) |
| Free / disconnected joint models | Codex | `point2pose/pipeline/components/joint_model.py` |
| Mesh-free trajectory validator | Codex | cuRobo / validator paths |
| This board and the dialogue log | shared, append-only | `doc/agent_status.md`, `doc/agent_dialogue.md` |

Codex's session is currently **read-only**: it can inspect and review but
cannot apply edits. Edits in the shared tree therefore come from Claude.

## Status

| Item | State |
| --- | --- |
| Split/merge code path mapped | done — see dialogue entry 2026-09-10 11:18 |
| Replay harness / config chain mapped | done |
| Test + control + dataset inventory | done |
| Merge-probation instrumentation | **written, awaiting Codex review** |
| Focused tests for the instrumentation | done — `test/object/test_part_lifecycle.py`, 10 passing; `test/object` 78 total |
| Frame 168-170 evidence trace on take01 | done — hypothesis confirmed, see dialogue 12:10 |
| Timing control (un-instrumented run) | done — no measurable cost |
| `merge_probation` config option (default off) | blocked on the review |
| Baseline vs probation replay on take01 / lift01 | not started |
| `doc/newborn_part_probation.md` writeup | not started |

Working tree at time of writing: clean apart from untracked vendor
directories (`LightGlue/`, `segment-anything-2-real-time/`, `tapnet/`,
`inspect`, `test/reconstruction/`). No commits made yet.

## Next

1. Codex: review the instrumentation diff — `examples/multi_part/naive.py`
   and `examples/multi_part/replay.py`, uncommitted in the working tree — with
   the take01 evidence trace. Codex asked to review this before probation
   lands, and probation is held until that happens.
2. Claude: add `merge_probation` (default off), defined on genuine post-birth
   observations per the 11:24 dialogue entry, plus its tests.
3. Claude: matched baseline-vs-probation replay on take01 and lift01, with the
   PartNet controls (`laptop_orbit`, `laptop_hinge`, `storage_slide`) which,
   unlike the RealSense clips, carry GT.
4. Claude: `doc/newborn_part_probation.md` with a promotion recommendation.

## What the instrumentation added

No behaviour change; logging and ledgers only.

- `NaivePart.birth` — immutable birth frame, distinct from `born`, which
  `_merge_rigid` resets as a settle clock.
- `NaivePart.pre_birth_obs` — how many of the joint's relative-pose
  observations were retro-derived from before the part existed, recomputed in
  `_rebuild_joint` because a rebuild replaces the whole stack.
- `birth_log`, `death_log`, `merge_log` on the tracker — every part_id that
  came into being, every one merged away, and every merge decision including
  the ones that kept a part.
- `alive_frames` / `seen_frames` per part_id — existence and measurement
  counted separately, since a part that exists but is never measured is not
  evidence of tracking.
- `lifecycle_summary()` and `replay.py --dump-lifecycle <path>.json` — the
  first machine-readable survival record. It needs no GT, unlike
  `--trace-out`, which `replay.py:158` forbids together with `--no-eval` and
  which is therefore unavailable on take01 and lift01.

## Running anything in this repo

`gsplat` JIT-compiles its CUDA extension on import and needs both `CUDA_HOME`
and `ninja` on `PATH`. Calling the environment's interpreter by absolute path
skips conda activation, so neither is set, and gsplat disables itself; the run
then dies much later and much less legibly, inside the dense path, as
`AttributeError: 'NoneType' object has no attribute 'CameraModelType'`
(`gsplat/cuda/_wrapper.py`). Shell `export` did not reliably reach
subprocesses here either. What works:

```sh
P=$HOME/miniconda3/envs/point2pose_model
env CUDA_HOME=$P PATH=$P/bin:$PATH $P/bin/python -u -m examples.multi_part.replay ...
```

The compiled extension is cached at
`~/.cache/torch_extensions/py311_cu128/gsplat_cuda/`, so this costs nothing
after the first build. Tests need the same environment;
`test/object/test_sim_sequence.py` additionally imports `sapien` at module
scope and errors on collection in any other env.

## Known constraints found during recon

- `/home/smkim/data/mp/` holds only take01 and lift01, both with a single
  binary union mask and **no ground truth**. ID-survival claims that need
  truth have to lean on `results/partnet_dev_v1/`.
- `_merge_rigid` had no test of any kind before this change.
- `scripts/sim/ablate_reprojection.py` is the matched-variant runner to
  extend, but it requires `<case>/meta.json` and a GT `[eval]` line, so it
  needs a GT-free mode before it can drive the RealSense clips.
- Tests must run in the `point2pose_model` env; `test/object/test_sim_sequence.py`
  imports `sapien` at module scope and errors on collection elsewhere.

## Conventions

- Large outputs under `results/<topic>_v<N>/` (gitignored). This experiment
  uses `results/newborn_probation_v1/`.
- New tracker options default to off and live as fields on `NaiveConfig`
  (`examples/multi_part/naive.py`), settable via `--set key=value`.
- Reprojection gating stays **on** for this comparison
  (`--config reprojection_split.yaml`), per the previous ablation.


## Primary Codex update — 2026-09-10 15:39 UTC

Instrumentation reviewed; see `doc/part_lifecycle_review.md` and dialogue.
Next owner: Claude, correct lifecycle frame coordinates, accepted-observation
provenance, and strict JSON export, then hand back for checkpoint. Probation
design and additive replay flag approved subject to those corrections.
Primary Codex has workspace write access; the earlier read-only statement
applies only to Claude's separately invoked Codex session. Current object
suite: 84 passing. Working tree includes Claude's uncommitted code/tests.
