# Recovery candidates supported by the wrong part

2026-09-10. Candidate snapshots now provide GT-free points, joint-grid poses,
depth, current object mask and calibration at each attempted recovery.
`recovery_debug_dir` is empty by default. `scripts/sim/diagnose_recovery.py` can
subsequently label depth-supported points using simulation GT. This is explicitly
an offline oracle diagnostic; GT labels never enter online recovery scoring.

## Observed failure

On the combined hinge/orbit retry run, the frame-110 winning candidate has
166 supported points, **all on the GT base and none on the lid**. Frames 109 and
111 likewise have all 173 and 160 supported points on the base. At frames 112,
114 and 115, base support is 145/146, 156/160 and 138/142 respectively. Thus
whole-object depth support is being confused with evidence for lid identity.
Snapshots and oracle report: `results/recovery_diagnosis_v1/candidates/` and
`results/recovery_diagnosis_v1/support.json`. Candidate matches precede temporal
confirmation; not every matching candidate is accepted online.

## Estimated ownership experiment

The optional `recovery_exclude_owned` removes pixels already depth-supported by
other currently observed estimated parts. It projects their stored Gaussian
centers at their estimated poses, accepts ownership only within 12 mm of measured
depth, and dilates these pixels with a 5x5 kernel. The remaining union mask is
used in the existing recovery scorer. No GT identity, pose or camera trajectory
is used. Hidden/misaligned geometry cannot claim depth pixels. The child is
excluded from its own ownership mask. Defaults remain unchanged.

This tests a standard geometric exclusivity cue, not a new correspondence or
articulation method. Misestimated other parts can still claim incorrect pixels;
dilation can suppress a nearby real child. The experiment does not construct
missing reverse-side geometry and requires broader controls before promotion.

Reproduce the online experiment with `ablate_reprojection.py --variants
guarded_recovery_ownership --trace`. Reproduce the offline diagnostic:

```bash
python scripts/sim/diagnose_recovery.py \
  --snapshots results/recovery_diagnosis_v1/candidates \
  --seq-dir results/partnet_dev_v1/laptop_hinge_orbit \
  --out results/recovery_diagnosis_v1/support.json
```

## Measured outcome

`results/recovery_ownership_v1` contains combined hinge/orbit and drawer runs.
Estimated ownership rejects all 11 combined-motion search results; the four
wrong accepted recoveries disappear. Independent lid observations remain 31.
At frames 110, 111, 112 and 115, position error changes from approximately
445, 444, 447, 445 mm to 75, 90, 116, 216 mm respectively. These latter poses are
**held, unobserved poses**, not successful recovery. The method prevents a
specific wrong-part snap but still cannot track the lid after observation loss.
Full values and flags are in `results/recovery_ownership_v1/frame_errors.json`.
Final first-pose-aligned medians stay unchanged and hide these transient failures.

Drawer retains two parts, 97.6% purity, 76 moving-part observations and 5.1 mm /
0.67 degrees aligned median error. It never invokes recovery, so it is a basic
non-regression control, not evidence that ownership preserves correct recoveries.
Step medians are 107.9 ms combined and 83.0 ms drawer (short-run timings only).
The option remains experimental/default-off. A positive recovery control is
required before promotion: tests of ownership alone cannot establish recall.
The relevant suite passes 64 tests, including depth-validated ownership and
label-count filtering. No novelty or generalization claim is made.
