# Audit: URDF export ↔ viewer consistency (read-only)

Author: Claude (demo worker), 2026-09-11, worktree `articulation-demo`.
Scope: root `examples/multi_part/urdf_viser.py`, `author_state.save_snapshot`,
`urdf_export.py`/`urdf_view.py`, against **31 real saved exports** under
`results/author_demo_live/*/` and the existing tests, at root revision
`ba4f821`. **Read-only; no edits made. No speculative fixes.** Existing viewer
left stopped.

## Verified correct (no defect)

- **Covariance / quaternion / scale conventions are internally consistent.**
  `GaussianCloud` stores quats **wxyz** (identity `[1,0,0,0]`,
  `gaussian_part_assignment.py:48`), `scales` as **metric std-devs**
  (`foot*scale_mult`, `:73`, not log), `opacities` linear `[0,1]`.
  `urdf_viser.covariance` reindexes `quats[:,[1,2,3,0]]`→xyzw for scipy and
  returns `R·diag(scale²)·Rᵀ` — the right covariance, PSD on all real exports
  (min eigenvalue > 0). Matches `test_gaussian_export_and_covariance`.
- **Root + chain transform reconstruction is exact for a well-formed
  JointModel.** `base @ link_poses(joints, root, q)['part1']` equals
  `base @ jm.at(q)` to **4e-10** for revolute/prismatic (verified at q =
  0.2/0.5/−0.3). `joint_frames`' `origin@motion(q)@offset == jm.at(q)`
  factorisation (incl. off-origin revolute pivot) holds. Root index →
  `base = poses[root_idx]` is consistent (link names are list-index `part{j}`,
  `poses` are list-indexed; confirmed on an export with root index 1).
- **Unassigned/hidden accounting is consistent**: `hidden = Σ(labels<0 or
  ≥nparts)`; the render loop only draws `labels==j` for `j∈[0,nparts)`, so the
  count matches what is withheld.
- All 32 tests in `test_urdf_viser.py` + `test_urdf_export.py` +
  `test_author_demo.py` pass.

## Defect 1 — viewer opens at q=0, never the saved configuration

`ExportViewer.q = [clip(0., lower, upper) for each joint]`
(`urdf_viser.py:45`). The saved articulation state is **not** restored; every
moving part snaps to `clip(0, lo, hi)`. For the real drawer
`20260910-141047-786237` (revolute, limits `[-0.274, 0.448]`, saved near
q≈0.17) the viewer opens ~10–34° away from where the object actually was when
saved, with no control to return to "as saved". `model.npz` records the true
per-part poses; the viewer discards them for non-root parts (see Defect 2).

Reproducer:
```python
from examples.multi_part.urdf_viser import load_export
data, joints, root = load_export('results/author_demo_live/20260910-141047-786237')
mov = [j for j in joints if j['type'] in ('revolute','prismatic') and j['upper']>j['lower']][0]
print(mov['lower'], mov['upper'])            # -0.274 0.448  -> q inits to 0.0, not the saved q
```

## Defect 2 — `model.npz` poses and the URDF joint chain disagree (undisclosed)

The demo's own viewer (`author_demo.draw`) places each part at its **raw
tracked** pose `model.npz['poses'][j]`. `urdf_viser` instead derives every
non-root part purely from the **joint chain** (`base @ link_poses(q)`),
**ignoring `poses[j]` for j≠root**. These two representations, shipped together
in every export, disagree by the **joint-model fit residual** — the tracker
tracks parts independently and fits the joint afterward, so at the save frame
`poses[child] ≠ poses[root] @ jm.at(q_saved)`.

Measured over real exports (best-fit q within the joint's own limits; star-root,
so parent = root — no chain confound):

| export | joint | rot resid | **trans resid** |
| --- | --- | --- | --- |
| 20260910-140837-314580 | prismatic | 0.1° | 0.8 mm |
| 20260910-140749-202433 | prismatic | 3.3°* | **22.4 mm** |
| 20260910-141047-786237 | revolute | 2.5° | **32.5 mm** |
| 20260910-123422-388350 | revolute (part2) | 5.6° | **37.6 mm** |
| 20260910-140950-857779 | prismatic (part2) | — | **31 mm** |

*A prismatic joint cannot represent rotation; a 3.3° residual on a prismatic
export means the joint **type** was fitted to a relative motion that also
rotates (a tracking/joint-fit quality issue surfacing in the export, not an
export-math bug).

Because `urdf_viser` trusts the chain and drops `poses[child]`, **the same saved
model renders parts up to ~4 cm apart in `author_demo` vs `urdf_viser`**, and
`metadata.json` discloses only "observed ranges, not verified mechanical
limits" — it does not state that the URDF rest geometry departs from the
recorded per-part poses by the fit residual. Reproducer: the two probe scripts
below (also runnable inline) reconstruct `base @ link_poses(q)` and compare to
`poses[child]`.

## Test gap that hides Defect 2

`test_viewer_fk_uses_saved_root_pose` builds its fixture as
`parts[1].pose = root @ joint.at(.2)` (`test_author_demo.state()`), i.e.
`poses[child]` is *defined* to equal `base @ jm.at(q)` with **zero** residual,
then asserts the viewer reproduces `base @ jm.at(.3)`. It therefore validates
`link_poses == jm.at` (correct) but **never** the real-data invariant that the
URDF chain agrees with the independently-tracked `poses[child]`. A test that
saved a fixture with `poses[child]` perturbed off the joint manifold and
asserted a disclosed bound (or that the viewer can reproduce the saved config)
would catch both defects. No such test exists.

## Not defects / notes

- The `have_gaussians = all(k in data for k in ('scales','quats','opacities'))`
  fallback to point-only + "Legacy point-only export" message is safe for
  older exports; all current exports carry the three keys.
- `load_export` succeeds on every real export (1-part and multi-part); empty
  parts (`labels==j` absent) are skipped without error.

## Bottom line

Two concrete, reproduced consistency defects — (1) the viewer cannot show the
saved articulation, and (2) the URDF joint chain and `model.npz` poses disagree
by the joint-fit residual (≤ ~5° / ~40 mm here), silently and untested — plus a
test that structurally cannot catch (2). The transform/covariance math itself
is correct. All findings are export/viewer-owner (Codex) territory; this is a
report only.
