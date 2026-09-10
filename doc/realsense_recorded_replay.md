# Recorded RGB-D replay diagnosis (2026-09-10)

Replayed all 321 frames of `/home/smkim/data/mp/take01` and all 504 frames
of `/home/smkim/data/mp/lift01`. Tracking is not yet reliable enough for a
manipulation demo. Both clips contain removable pieces/rings and changing
contacts, rather than a guaranteed permanently connected articulation tree.

## Reproduction and artifacts

Base commit: `99a41a81778945cd02e80a9a33da5234d9d7f144`. Existing `masks.npz`
was used; masks were not regenerated with SAM2. No GT part labels or poses
were supplied. Outputs and logs are local, ignored by git, under
`results/realsense_recorded_v1/`.

```sh
python -u -m examples.multi_part.replay \
  --seq-dir /home/smkim/data/mp/take01 --method naive \
  --config reprojection_split.yaml --stride 1 --view 0 --hyp-panel 0 \
  --no-eval --vis debug \
  --out results/realsense_recorded_v1/take01.mp4 \
  --dump-joint results/realsense_recorded_v1/take01_joints.npz \
  --save-model results/realsense_recorded_v1/take01_model.npz
```

Repeat with `lift01` paths. The third run uses take01 and
`--set split_reprojection=false`, output stem `take01_no_split_gate`, without
`--save-model`. `run_metadata.json` records the cases; `input_quality.json`
and `mask_io_benchmark.json` record input checks and the reader benchmark.
Videos and `*_tracking_samples.jpg` allow qualitative inspection. The saved
models contain estimates, not validated manipulation-ready geometry or URDFs.

## Observations and ablation

| Run | Split frames (resulting part count) | Final parts | Depth recovery |
| --- | --- | --- | --- |
| take01, current | 168 (2), 230 (2) | 2 | 0 |
| take01, no split reprojection gate | 168 (2), 227 (2), 257 (3) | 3 | 0 |
| lift01, current | 104 (2), 127 (3), 184 (4), 340 (5), 410 (6) | 6 | 0 |

In take01, the blue piece visibly moves before its first split at frame 168.
That new part merges back at frame 170 in both runs. The baseline reports
179 coassociation split refusals for inadequate shared histories. This points
to discovery/history and early merge behavior as concrete investigation targets.
Disabling the reprojection gate does **not** advance first discovery and later
adds a qualitatively dubious extra revolving part. Simply relaxing reprojection
is not supported by this control; retain the current default.

Lift01 separates several relative motions, but sampled overlays show unstable
poses/axes and lost tracks while groups of rings are lifted. Six parts alone
does not establish oversegmentation: there are multiple physical pieces and no
GT correspondence. The final fitted joints include sliders spanning 435–753 mm
and hinges, including one with confidence 0.03. These fits are not evidence of
permanent mechanical joints. Current joint selection has `allow_free=false`;
a detachable piece can be forced into an inappropriate constrained model.

Median valid depth inside the supplied mask is 94.5% for take01 and 96.7% for
lift01 (minimum 92.1% and 91.0%). This rules out pervasive missing depth as a
complete explanation, but does not establish depth accuracy, RGB registration,
or temporal mask correctness. Low texture and hand occlusion remain plausible
contributors; this experiment does not isolate their effects.

Tracker step medians were 99.4 ms (take01), 181.2 ms (lift01), and 117.8 ms
(no gate). These are not end-to-end live rates or a controlled timing ablation:
the latter runs overlapped, and the mask reader changed after the first run.

## Reader fix and remaining work

Compressed NPZ arrays cannot be memory mapped by `mmap_mode='r'`. Previously
every mask query decompressed the entire stack. `Recording` now decodes once,
closes the archive, and returns cached frame views. For 30 take01 queries,
repeated decoding took 2.143 s versus 0.104 s initialization and 0.000011 s
cached queries. This fixes offline I/O, not the live tracking failure. Memory
remains one full mask stack (about 307 MB per 1000 uint8 640×480 frames).

Next tracking experiment: instrument newborn-part merge decisions and test a
bounded probation period with matched replay, measuring ID survival and false
parts rather than only final count. Separately distinguish unconstrained part
motion from evidence for a lasting joint; changing contact needs a free or
disconnected hypothesis before URDF export can be trusted. These are proposed
follow-ups, not implemented fixes or novelty claims. Keep reprojection and
compare on a permanently attached hinge/slider recording as well as these clips.
