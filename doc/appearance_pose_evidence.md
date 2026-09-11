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
