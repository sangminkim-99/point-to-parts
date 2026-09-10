# Observed coverage and shared-frame pose diagnostics

2026-09-10. `scripts/sim/compare_pose_traces.py` evaluates saved traces only;
none of its GT information is passed to the tracker. It requires identical frame
indices, GT names and GT trajectories across runs and rejects mismatches.

A candidate must be present, marked observed, have a finite pose, and carry at
least eight non-background seed-label votes with at least 60% dominant-label
purity. For each GT group/frame, select the candidate with most matching votes
(tie: persistent ID), never lowest pose error. Report duplicate candidates,
selected IDs, switches between successive observations, first qualifying frame,
missing frame indices and coverage. These are seed-label-based associations:
a mixed initial part following lid motion need not qualify as a lid. Missing
coverage means no qualifying observed association, not necessarily no pose.
Thresholds are evaluation settings, not calibrated tracking confidence.

Each selected ID/GT pair receives one fixed gauge alignment at its first
qualifying observation in that run. Accuracy is then evaluated both on its own
observations and on the intersection shared by every compared run. There is no
new alignment at the shared-window boundary. An empty intersection yields null
accuracy, not zero error. Held poses are missing coverage. A new ID has a new
alignment; the explicit ID/switch counts must accompany its error. This evaluates
motion after association, not absolute reconstruction or discovery accuracy;
initial alignment frames can differ between runs. Do not interpret it as a fully
matched absolute pose benchmark.

## Measured results

All traces have 119 evaluated frames. Output JSON and full missing-frame lists
are in `results/pose_common_v1/{hinge_orbit,hinge,slide}.json`.

| Clip / group | Compared runs | Shared frames | Shared median mm / degrees | Observed coverage |
|---|---|---:|---|---|
| Hinge + orbit / base | Reprojection reference; guarded | 41 | 20.61 / 2.54; 9.61 / 0.64 | 62/119; 41/119 |
| Hinge + orbit / lid | Reprojection reference; guarded | 0 | unavailable for both | 0/119; 31/119 |
| Clean hinge / lid | Unconditional incremental; guarded | 44 | 430.10 / 56.79; 434.25 / 43.00 | 53/119; 47/119 |
| Drawer / moving group | Unconditional incremental; guarded | 76 | 11.38 / 1.19; 5.11 / 0.67 | 77/119; 76/119 |

The guarded combined-motion lid first qualifies at trace frame 79, remains
observed through 109, and is missing from 110–119. Its own-observed median is
29.04 mm / 4.48 degrees. Earlier reporting counted 41 pose entries and 36.5 mm /
4.65 degrees, including held poses. Neither metric establishes continuous lid
tracking. The reference has duplicate base candidates on 62 frames and no
qualifying lid association. Do not report a comparative lid accuracy gain.

The clean-hinge common window confirms that both incremental variants remain
bad (over 430 mm lid error), despite a lower rotation error for guarded. The
slider favors guarded in the common window, consistent with the earlier result;
guarded matches the existing reprojection baseline there. No default changed.
These are diagnostic results from one object per case, not generalization claims.

## Reproduce

```bash
python scripts/sim/compare_pose_traces.py \
  --trace baseline=results/pose_diagnostics/baseline.npz \
  --trace guarded=results/pose_guarded_v1/laptop_hinge_orbit/guarded_incremental/trace.npz \
  --out results/pose_common_v1/hinge_orbit.json
```

Tests cover non-realignment on subset selection, explicit missing coverage,
selection by label evidence rather than best pose, ID switches/duplicates, empty
intersections and rejection of different GT trajectories.
