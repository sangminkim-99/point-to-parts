# Scan-first, then-articulate: gray/disappearing surfaces — diagnosis

Author: Claude (tracking worker). Worktree `wt-articulation-probation`. Status:
**trace run complete; it REFUTES the split-misassignment hypothesis. No
behaviour changed; nothing promoted.** Read §5 (trace results) for the finding;
§1–§4 are the read-only groundwork that led to it.

Symptom (user): show the whole box first, then move the lid. Only the currently
visible lid/body regions separate; the rest of the scanned surface goes gray /
disappears.

**Headline finding (§5–§6):** the 40% of the dense cloud that ends up unavailable
(−1) is **grow-then-carve churn during the moving-object scan**, not
visibility-blind misassignment at the split (that path is ~1% of carves; the
split assigned all 14,086 held gaussians with close seeds). And the churn does
**not** lose the *currently observed* surface: coverage stays complete (uncovered
≈ 0, ending 0.000) while the alive count grows 10,235 → 15,180. So the −1 are
superseded stale copies, and the live reconstruction of what the camera sees is
never holed. The prescribed labelling fix is **not** the lever. The user's symptom
(faces going gray after they stop being observed) points at a *different*
question this run does not answer: **retention of previously-scanned faces that
are no longer observed** — current-camera coverage cannot see them. That, or a
2D-overlay/viewer effect, is the real next measurement.

The manager asked, correctly, to first establish **whether the gray is a 2D
sparse-track overlay or real 3D geometry**, and not to name a mechanism from the
snapshot: the saved −1 proves geometry is *unavailable*, not the temporal cause.
Four hypotheses were kept open until the trace: (a) initial misassignment,
(b) coordinate-frame mismatch, (c) carve after wrong assignment, (d) display-only
gray. The trace refutes (a) as the driver and keeps (b) open (§5).

## Evidence so far (read-only)

### 1. What the saved model proves — and what it does not

The latest saved live model (`results/author_demo_live/20260910-123857-878059/
model.npz`, 2 parts, frame 627) has this label histogram over 25,033 gaussians:

```
label -1 :  9,945   (40%)   unavailable geometry
label  0 :  4,422   (18%)   body
label  1 : 10,666   (43%)   lid
```

**What this proves:** at save time, 40% of the cloud carried label −1, and the
author viewer (`author_demo.py:144`) renders only `labels == j` per part, so −1
geometry is not drawn — that much scanned surface is **unavailable** in the
saved model.

**What this does NOT prove** (and I previously overstated):
- It is a single end-state snapshot, so it does **not** establish the *temporal
  cause* — when or why those gaussians became −1.
- It does **not** establish that this dense −1 is what the user meant by "gray".
  There is also a 2D gray in the RGB panel (`naive.render` draws sparse tracks
  with `owner < 0` in gray; `streaming.py` labels "grey object-explained-by-no-
  part"). Which of these the user is describing is not settled from the save;
  it needs their view or the annotated trace.

So the honest statement is: **there is real unavailable dense geometry (40%);
its cause and its identity with the user's "gray" are open** until the trace.

### 2. Hypotheses still open (nothing ruled out without the trace)

The four hypotheses — (a) initial misassignment, (b) coordinate-frame mismatch,
(c) carve after wrong assignment, (d) display-only — are **not** decided by the
snapshot. In particular I withdraw the earlier "coordinate mismatch is unlikely"
call: the code reads as though sparse `anchor_xyz` (`naive.py:410`) and dense
`cloud.means` (`dense_model.py:15`) share the anchor frame, and the saved model
looks spatially coherent, but **coherence at save time does not rule out a
transient frame error during the split**, and that is exactly what a per-split
trace shows. (b) stays open until the trace.

### 3. The code path — a hypothesis to test, stated carefully

`_split_labels` (`naive.py:2073`) reassigns **all** held gaussians to whichever
child's *currently-tracked sparse seed* is nearest in anchor space:

```python
held = np.where(self.model.labels[:len(gm)] == j)[0]   # ALL parent gaussians
seeds = [anchor_xyz[p.idx[anchor_ok[p.idx]]] for p in new]   # sparse tracks only
_, nn = cKDTree(pts).query(gm[held], k=1)
self.model.labels[held] = own[nn]
```

This is visibility-blind, so a *hypothesis* worth testing is that historical
surface with no nearby live seed is assigned to the wrong child (a). But this is
a code-reading, **not measured**: the seed distances the trace records are what
confirm or refute it.

`carve` (`dense_model.py:79`) is the only writer of −1, and it is **narrower than
I described**:

```python
carve = (observed_depth > projected_z + margin)   # gaussian is IN FRONT of surface
free  |= carve & (labels == j)                     # only under its OWN part's pose
```

Correction: carve fires **only when the gaussian projects in front of the
observed surface** (`zo > z + margin`) — genuine free space. A gaussian that is
merely **occluded** (`zo < z`, something nearer in front) is *not* carved — so
occlusion is already protected, and it is **wrong** to say "not visible is
treated as contradiction". The failure mode that remains possible is narrower: a
gaussian assigned to the *wrong* part, projected under that part's pose, can land
in front of the observed surface and be carved (c after a). Whether that actually
happens — and how much of the 40% it accounts for versus ordinary stale-copy
carving as `grow` adds and `carve` removes — is precisely what the trace decides.

### 4. What the static snapshot cannot settle

The −1 points are interleaved with survivors across the whole object (within
3–6 mm), not one clean removed face — consistent with misassign-carve, stale-
copy carve, or a mix. Separating them needs **temporal provenance**: per-split
seed distances and per-carve attribution (was the carved gaussian assigned via a
far seed, or never split-assigned?). That trace is the deciding evidence, below.

Corroboration across the two saved live sessions (not a controlled comparison —
different recordings — but it shows the −1 fraction is large and variable, not a
fixed overlay):

```
20260910-123422-388350  3 parts, frame 803:  −1 = 1,786  (10.7%)
20260910-123857-878059  2 parts, frame 627:  −1 = 9,945  (39.7%)
```

## 5. Trace results — the deciding evidence (and it refutes hypothesis (a))

The instrumented replay ran to completion (build fixed per the user's env; built
into a private `TORCH_EXTENSIONS_DIR` so the shared gsplat cache was untouched).
Staged `capture-20260910-123622-842007` (symlinked rgb/depth + a rebuilt
`masks.npz`, since the capture stores PNG masks and `Recording` reads
`masks.npz`). Artifacts: `results/surface_ownership_v1/capture_baseline_*`.

Faithful reproduction — the run reaches the same end state as the saved model:

```
final labels: -1 = 9,988 (39.7%),  part0 = 4,332,  part1 = 10,848   (25,168 total)
grown 14,933   carved 9,988   splits: 1 (at frame 196)
```

**Split provenance (the one split, f196, parent 0 → two children):**

```
held = 14,086   assigned 6,961 / 7,125
seed distance to the assigning sparse seed:  p50 1.1 cm,  p90 3.5 cm,  max 8.1 cm
held with seed > 5 cm: 139 of 14,086  (1.0%)
```

The split assigned **all** held gaussians — not just the visible ones — and did
so with **close** seeds. By frame 196 re-seeding had populated sparse tracks
densely across the scanned surface, so almost nothing was assigned "blind to a
far seed". **Hypothesis (a) — visibility-blind misassignment at the split — is
not the driver.**

**Carve provenance (9,988 carved across 161 carve events over the whole clip):**

```
carved via a FAR split-seed (>5 cm):     105   ( 1.1% )   <- the (a)->(c) path
carved via a split-seed >2 cm:         2,142   (21.4% )
carved never-split-assigned (stale):   3,729   (37.3% )   <- grown/initial copies
remainder: split-assigned with a CLOSE (<2 cm) seed, carved anyway
```

So the 40% loss is **not** guess-then-carve. It is dominated by **grow-then-carve
churn during the moving-object scan**: 14,933 gaussians were grown as the object
was moved to show all sides, and 9,988 were later carved because, projected under
their part's pose, they landed in front of the currently-observed surface. Only
~1% of carves trace to a far-seed split misassignment. The single split assigned
everything with close seeds.

**This overturns my earlier "leading explanation".** The prescribed fix
(visibility-aware historical *labelling at the split*) would touch ~1% of the
loss. The real target is the **grow/carve behaviour under whole-object motion**:
why confidently-assigned, correctly-scanned surface is carved once the object
moves. Two candidates, and separating them is the next measurement, not a claim:
- *Legitimate stale removal* — a gaussian grown at an earlier object pose is a
  genuine duplicate once the object moves, and carving it is correct; the model
  simply re-grows the surface at the new pose, so 40% "churn" may be mostly
  healthy turnover that only *looks* like loss in a single end-state snapshot.
- *A frame/pose problem in grow or carve* — if grown gaussians do not track with
  their part (grown in the wrong frame, or carved under a drifting pose), correct
  surface is destroyed. This is exactly why coordinate-frame (b) must stay open:
  the trace shows confident-seed geometry being carved, which a frame error would
  also produce.

## 6. Coverage-over-time — the observed surface is never lost

Re-ran the same capture in an isolated gsplat cache (`results/torch_extensions`,
so the live cache was untouched) with a default-off per-frame coverage log
(`uncovered` = fraction of the *currently observed* object surface no part
explains; plus alive/dead gaussian counts). Deterministic reproduction (same
39.7% −1, 9,988 carved, 1 split). Result:

```
uncovered fraction:  mean 0.025   p50 0.001   p90 0.088   max 0.351   last 0.000
alive gaussians:     first 10,235  ->  last 15,180        (grew, did not shrink)
dead (-1):           0  ->  9,988   (monotonic)
first-half uncovered 0.023  ~  second-half 0.026          (no degradation)
```

**The currently-observed surface is essentially always fully covered** (uncovered
near 0, ending at 0.000) and the **alive count grows** (10,235 → 15,180) even as
9,988 are carved. So for the surface the camera is looking at, **re-grow more than
replaces carve — there is no loss of observed surface at any point.** The 40% −1
is superseded stale geometry (old-pose copies), not a hole in the live
reconstruction of what is seen. This makes the *legitimate stale-removal* reading
the correct one for the observed surface, and argues against a gross grow/carve
frame bug for it.

**What this does NOT settle — and it is the crux of the user's symptom.**
`uncovered` measures coverage of the **currently observed** surface only. It does
**not** measure whether a face scanned *earlier* but not observed *now* is still
present in the 3D model. The user's report — show the whole box, then move the lid,
and "the rest" goes gray — is about faces that are no longer being observed. This
metric cannot see them, so it neither confirms nor refutes their loss. Two things
remain open and are the real next questions:
- *Retention of non-observed scanned faces.* Does carve (under motion, or a
  slightly drifting part pose) remove a previously-scanned face that is currently
  edge-on/averted, without re-grow replacing it (grow only adds where the
  **current** view is uncovered)? That would be invisible to `uncovered` and
  visible to the user rotating the 3D view. This needs per-face / multi-viewpoint
  coverage, not current-camera coverage.
- *Which "gray" the user means.* With observed-surface coverage complete, the
  reported gray may instead be the 2D sparse-track overlay (`owner < 0`) or the
  viewer dropping −1, rather than missing observed geometry. Confirming this needs
  the user's view or an annotated frame.

## Next steps

1. Measure **retention of non-observed faces** (multi-viewpoint coverage of the
   final model against the union of all scanned surface), since current-camera
   `uncovered` is already ~0 and cannot see this. This is the measurement that
   matches the user's symptom.
2. Only then choose a fix. If it is churn/display, the fix is likely on the
   *viewer/model* side (show pending/uncertain instead of dropping −1; or retain
   a per-part canonical cloud) — the demo worker's display area, coordinated. If
   it is a frame/pose bug, the fix is in grow/carve. The visibility-aware
   labelling design below remains valid for the *split-assignment* quality but is
   **not** the lever for this 40%.
3. Keep probation and all current defaults intact; Codex reviews before any
   change. Nothing here is promoted.

## Appendix — visibility-aware split labelling (design; NOT the fix for the 40%)

**Read §5 first.** The trace refuted guess-then-carve as the cause of the 40%
loss (only ~1% of carves trace to a far-seed misassignment; the split assigned
everything with close seeds). This design therefore is **not** the fix for the
reported symptom. It is retained only because it still improves *split-assignment
quality* (the ~21% of carves with a >2 cm seed, and future captures where seeds
are sparser than in this one), and because the manager originally asked for it.
It must not be presented as addressing the gray surfaces. Two changes, default-
off, tested as matched controls:

**A. Visibility-aware historical labelling (`_split_labels`).**
Instead of assigning all held gaussians blind:
1. *Confident seeds only.* Assign a held gaussian to a child only when it has a
   nearby confident seed — `seed_dist < split_seed_radius` (the distances this
   diagnostic already records) — AND, where the gaussian was observed in recent
   keyframes, the child's motion hypothesis reprojects it consistently with the
   stored depth/mask (multi-keyframe depth/mask reprojection under both child
   motions, with the correct anchor transforms and occlusion handling).
2. *Spatial propagation.* Grow confident labels to neighbours along the surface
   (kNN + normal/topology continuity), so a confidently-owned region extends to
   its own surface rather than to the nearest sparse dot.
3. *Revisitable hypothesis for the rest.* A held gaussian with no confident seed
   and no consistent reprojection is left **pending** (a distinct label, not a
   part and not −1), carried across frames and resolved later when the surface
   is observed again. It is neither guessed into a part nor deleted.

**B. Carve is already occlusion-safe; the change is narrower.** Carve does *not*
delete occluded gaussians (`zo < z` is protected); it deletes only gaussians in
front of the observed surface. So B is **not** "don't carve invisible points" —
that is already true. The only addition worth testing is: **do not carve a
gaussian while its owning part's pose is not confidently observed this frame**
(a low-confidence pose can project a correct gaussian into false free space).
And crucially, a **pending** gaussian (from A) is carved by no part's pose, since
it belongs to none. If the trace shows the 40% is mostly misassign→carve, then
fixing assignment (A) is the lever and B is a small backstop; if it is mostly
stale-copy carve, neither A nor B is the fix and the target is grow/carve
bookkeeping instead. **The trace decides which.**

**Provenance.** Every assignment and every resolution of a pending gaussian is
logged with its evidence (seed distance, which keyframes, which motion), so the
association is auditable — the same discipline as the probation ledger.

**Matched controls.** Replay each saved capture (scan-first, then moving lid)
under baseline vs A vs A+B, reporting: final label histogram (−1 fraction),
retained-surface fraction, and — where a GT-bearing sim scan-then-articulate
sequence exists or can be rendered — per-part IoU of the retained geometry
against GT, so "retained more" is shown to be "retained *correctly* more", not
just "carved less". Nothing promoted to default without that evidence and
Codex's review.
