# Appearance evidence baseline

`appearance_evidence.py` measures normalized RGB L1 error at projected stored
surface centres under an explicit pose. It keeps the nearest centre per pixel
and requires current object-mask/depth support. Hidden, offscreen and invalid
samples give no positive evidence. The sample count is always returned; an
empty projection returns no score rather than a perfect score.

This is a diagnostic candidate-scoring component, not Gaussian rasterization,
not an optimizer, and not wired into tracker acceptance. Candidate scores with
different support sets cannot be compared blindly: losing hard samples can
artificially reduce error. Matched support/coverage checks and illumination
robustness are required before promotion. Stored colors must be normalized
RGB; observations are uint8 RGB. Front/back appearance differences may reject
a mirror pose but cannot by themselves produce a correct unseen-face pose.

Next experiment: use the symmetric and face-distinct turnover controls,
freeze pre-flip geometry/colors, and report support count and RGB error under
actual tracker candidates. Any GT-pose comparison is an explicitly oracle
reference only; it cannot be used for runtime candidate generation.

`paired_appearance_evidence` now intersects the visible stored-sample IDs under
both candidates and scores only that intersection. It reports both individual
sample counts and the paired count. Fewer than 20 shared samples gives no score
(configurable for experiments), not zero loss. A regression test drops a hard
sample out of the second view and verifies that this produces no paired gain.
This removes one selection artifact; it does not certify the discarded regions
or solve illumination changes, occlusions, or front/back ambiguity.

## First diagnostic results (September 11)

Worker artifacts: `results/thin_turnover_v1/appearance_rigid/appearance_probe.json`
and `appearance_rigid_distinct/appearance_probe.json`. Both are 120-frame
simulation controls with geometry/colors frozen at frame 16 (about 4,983
centres), oracle union masks, and an explicitly oracle GT-pose comparison.

Post-flip median RGB L1 under the wrong tracked pose is about 0.000014 for
uniform faces and 0.289 for distinct faces. This suggests appearance can expose
an inconsistency in the distinct-face case. It does not establish a threshold
or a correct pose candidate.

The GT-pose diagnostic scores about 0.353 (uniform) and 0.347 (distinct), with
only 363 supported centres versus roughly 4,950 for the tracked pose. It is
therefore not a matched comparison and cannot justify ranking either candidate.
The stored model has never seen the bottom face; lighting changes also affect
color. Paired-sample evaluation and a pre-flip error distribution have been
requested. Keep scoring diagnostic-only until these confounds are resolved.

The paired follow-up (`appearance2_rigid*`) still favors the wrong pose on all
54 post-flip frames with enough common samples. Distinct-face paired median
errors: tracked 0.183, oracle 0.346 (362 common centres). This is a failed pose
ranking result, not evidence to promote an appearance-based selector.

Viser now displays RGB MAE from the already-produced approximate composite,
using only raw camera RGB and depth-supported object pixels. The pixel count
is shown alongside the value; annotated tracking overlays are never scored.
This metric is display-only and does not establish pose correctness. It adds
no rasterizer calls and remains subject to lighting and composite artifacts.
