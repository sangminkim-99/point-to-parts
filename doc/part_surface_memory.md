# Persistent part identity across changing visible surfaces

## Objective and scope

Keep a lid as the same physical part when its outer face disappears and its
inner face becomes visible. This work does not change SAM2 or try to exclude
manipulators. Reprojection is the association mechanism; matching the appearance
of the front to the appearance of the back is not required.

The experimental preset is `configs/multi-part/surface_memory.yaml`. This is a
step toward robust tracking, not a claim that arbitrary unseen back surfaces or
long complete occlusions are solved.

## Implemented method

1. **Detect missing surface coverage.** Projected, reliable image tracks cover
   only part of the current RGB-D object mask. Sample connected uncovered regions
   even when the main body's track count remains healthy. Previously a well
   tracked box prevented fresh points from being sampled on its opening lid.
   These fresh observations receive no assumed co-motion prior with the main
   body; persistent motion and the existing split tests must establish ownership.
   New-face discovery currently enables this coverage trigger while the object
   is still a single part. Established parts keep their own-region reseeding.

2. **Separate physical identity from observation identity.** A part has a
   persistent ID and canonical coordinates. New image tracks and Gaussians can
   be added to that part; they do not define a new physical identity. A split
   retains the original ID for its larger group and gives the other group a new
   ID. Inserting another part does not renumber the existing identities or their
   display colors. This does not constitute general semantic re-identification.

3. **Recover a known joint by RGB-D reprojection.** With a reliable parent pose
   and an already fitted revolute/prismatic joint, generate candidate child poses
   by sampling joint state. For each canonical point X of the child:

   `T_child(q) = T_parent * JointModel.at(q)`

   `u(q) = project(K * T_child(q) * X)`

   Compare the predicted camera depth at u(q) with observed depth and the object
   mask. Score supported surface points, penalize predicted geometry in observed
   free space, and treat geometry behind a foreground observation as potentially
   occluded. The implemented score is:

   `score = (supported_count - 0.5 * free_space_count) / model_point_count`

   Default depth tolerance is 12 mm. Require at least 20 supporting model points,
   a score of 0.3, and a margin of 0.05 over a geometrically distinct alternative.
   Neighboring samples that move median model points less than 24 mm are treated
   as the same noisy pose for this ambiguity check. At most 768 model points and
   121 joint-state samples are used by the tracker integration.

4. **Confirm before attaching the new view.** Require compatible recovery on
   two consecutive frames. Supported reprojected pixels define a reseeding
   region. Backproject new points and store `X_new = inverse(T_child) * x_camera`
   under the existing part ID. The joint search is correspondence-free: a thin
   panel's opposite face can agree geometrically despite having different color
   and entirely different image feature IDs. This works only to the extent that
   the stored surface approximates the newly visible surface within tolerance.

5. **Keep predictions out of model learning.** If sparse fitting fails, retain
   the last pose for identity/display but mark it unobserved. Do not append that
   held pose to joint history or let dense ICP silently make it a measurement.
   A recovered joint state is not fed back as independent evidence for fitting
   that same joint's axis/type. Skip free-space carving for unobserved parts and
   freeze growth while any part remains unconfirmed, so the main body does not
   absorb the missing part's new surface during that interval.

6. **Require adequate historical support.** Retrospective fitting must observe
   the configured fraction of the entire discovered part's tracks, not merely
   the subset whose indices existed in an old frame. Otherwise three surviving
   old tracks can claim 100% support for a part mainly discovered from a later
   face, contaminating the learned joint history.

There is no differentiable RGB renderer optimizer in this change. The normal
tracking path remains sparse 3D registration and projective depth ICP. The new
fallback evaluates joint-constrained geometry by image reprojection.

## Validation

Sixteen tests pass across `test/object/test_surface_memory.py` and the existing
part-splitting tests. New tests cover:

- newly exposed surface despite healthy coverage on the main body;
- a panel viewed at 140 degrees, with a 3 mm opposite-face offset and no shared
  image correspondences;
- partial occlusion, complete occlusion, missing depth and ambiguous alternatives;
- temporal confirmation, persistent identity, and no feedback into joint fitting;
- rejection when the parent pose is stale;
- part-ID stability when the part list changes;
- one fixed trajectory-frame alignment that preserves subsequent drift;
- rejection of retrospective poses backed by too little of the discovered part.

These synthetic recovery tests assume the hinge is known correctly. The RBO
results below separately test discovery with no GT part labels in the input.

### RBO cardboardbox01_o

Input masking is unchanged from the earlier experiment: SAM2.1 Hiera Large,
one initial GT-derived object bounding box, then causal propagation. No added
negative prompts or manipulator masks. Existing GT is used only for initial
prompting/frame selection and evaluation, not as part labels for the method.

| Run | Frames | Discovered parts | Sparse-label purity | Median step FPS |
|---|---|---|---|---|
| Previous default | 0–239 | 1 of 2 | 50.0% | 15.5 |
| New-surface preset, initial experiment | 0–239 | 2 of 2 | 98.9% | 10.6 |
| Persistent IDs/reprojection integration, before historical-support fix | 0–408 | 2 of 2 | 95.4% | 9.2 |
| Final, including historical-support fix | 0–408 | 2 of 2 | 95.4% | 9.5 |

The new preset also uses a six-frame track grace, a 600-track cap and
retrospective fitting, so this is a configuration-level result rather than a
single-switch ablation. The first split is at internal frame 110. On the full
run above, there are no further splits or merges: IDs 0 and 1 survive through
closing. The final joint is revolute, with evaluator axis error 9.1 degrees and
axis-line distance 3 mm. However, joint state/range is not trustworthy: the
reported excursion is 276 degrees, and the early confidence label calls the
joint prismatic. Retaining identity alone does not solve pose accuracy.

The full run above has zero depth-only recovery activations: ordinary tracks
remain sufficient to produce sparse fits, even when those fits drift. It tests
the new-face sampling and identity path, not real-data validation of the
correspondence-free recovery fallback. That fallback is validated synthetically.

After a single alignment at the first available predicted pose, median trajectory
errors in that run are 11.8 mm / 1.32 degrees for the base and 79.1 mm / 8.28
degrees for the lid. These measure subsequent tracking, not absolute initial pose
accuracy. The old frame-0 metric mixes canonical-frame offset with tracking error
for parts anchored later; it is retained in logs for comparison.

FPS measures tracker/reconstruction steps and excludes SAM2, disk I/O, evaluation
and display. Videos are written at fixed playback rate, not measured live rate.

The final supported-history run also retains IDs 0 and 1 to source frame 408,
with one split at internal frame 110 and no later splits/merges. It uses joint
tracking on 19 part-frames, but still has zero depth-only recovery activations.
The final revolute axis error is 7.5 degrees with 17 mm axis-line distance;
reported excursion drops to 163 degrees. This is not uniformly better geometry:
the lid's aligned trajectory errors remain 80.4 mm / 8.47 degrees, and the early
classification is still prismatic. Median/p90 step times are 105.2/159.8 ms.
Final artifacts use the prefix `full_supported` in the local output directory.

## Run and inspect

With the environment/CUDA setup in `doc/rbo_sam2_demo.md`:

```bash
python -u -m examples.multi_part.replay \
  --seq-dir /home/smkim/workspace/dataset/RBO/sequences/cardboardbox01_o \
  --method naive --config surface_memory.yaml \
  --object-masks results/rbo_surface_memory/cardboardbox01/full_masks.npz \
  --stride 1 --view 0 --hyp-panel 0 \
  --out results/rbo_surface_memory/cardboardbox01/replay.mp4 \
  --dump-joint results/rbo_surface_memory/cardboardbox01/joints.npz \
  --save-model results/rbo_surface_memory/cardboardbox01/gaussians.npz
```

Local artifacts are in `results/rbo_surface_memory/cardboardbox01/` (gitignored).
The snapshot includes persistent `part_ids`; Gaussian labels and parent indices
still index the snapshot's ordered part list. Model snapshots are not complete
resumable tracker checkpoints.

## Remaining work

- Trigger recovery on weak/inconsistent sparse fits as well as complete fit loss,
  using independently validated reprojection evidence.
- Maintain a consistent multi-view surface map and refine the shared hinge and
  part poses across keyframes; avoid treating uncertain early joint estimates as
  authoritative. The false early confidence and excessive excursion remain
  visible failure modes.
- For thick parts or entirely unseen geometry, preserve alternative pose/view
  hypotheses instead of assuming a thin-panel opposite face. If the evidence is
  indistinguishable, request more motion rather than claim identity certainty.
- Measure this on additional sequences and camera motions; the current real-data
  experiment is a single box sequence.

Design references for maintaining posed observations in a shared object map:
[BundleTrack](https://arxiv.org/abs/2108.00516) and
[BundleSDF](https://bundlesdf.github.io/). The contribution implemented here is
the part-specific coverage trigger and joint-constrained reprojection fallback;
their full keyframe optimization systems are not implemented or reused here.
