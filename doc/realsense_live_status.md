# RealSense live tracker verification

2026-09-10. A connected D435 was detected and aligned 640x480 RGB/depth frames
were captured successfully (84.8% valid depth in one initialization sample).
The existing `live_acquire` uses the current NaivePartTracker/configuration,
SAM2 whole-object mask and TAPIR. The user selected an object in the GUI.

The session initialized SAM2 with one box and TAPIR with 200 points, processed
through frame 287, and exited normally. Part splitting was reported at frame
107 and a revolute diagnostic at frame 114. Tracker-step throughput was reported
as about 8.4 FPS; SAM2 time is outside that timer, so this is not live pipeline
throughput. There is no GT trajectory or shape reference for this session, so
joint accuracy or success of manipulation cannot be inferred from these logs.
No RGB-D recording/model artifact was saved by this live entry point.

Start from the point2pose_model environment with the CUDA variables in the main
handoff document:

```bash
python -u -m examples.multi_part.live_acquire \
  --config reprojection_split.yaml --bbox-prompt --depth 1
```

Drag around the whole object; S initializes tracking, R resets, D toggles the
depth display, Q exits. Experimental recovery options remain off. The camera
requires host device access. Qt printed missing-font warnings but the session
initialized and tracked successfully. The ended session was not restarted
without the user being ready to interact.

## Live diagnostic correction

Previously the acquisition banner latched READY forever after first reaching its
confidence threshold, even if tracking or confidence subsequently failed. It
could also accumulate the streak from different joints. Now current readiness
requires consecutive qualifying frames of the same persistent ID and joint type,
fresh child and parent observations, valid finite confidence and excitation.
Identity/type changes, frame gaps, observation loss and mask loss reset the
streak; reacquisition requires the full hold duration. `ready_at` remains a
historical metric, distinct from current `ready`. UI says ESTIMATE STABLE and
axis RMS; no controllability claim is made. The rest of the tracking pipeline
is unchanged. Regression tests cover revocation, reacquisition, identity/type
changes, gaps, stale parents, invalid confidence and non-articulated models.

These changes apply on the next launch; they were not hot-loaded into the ended
session. Next useful live work: optional RGB-D/mask/pose recording for repeatable
failure analysis and point-cloud/joint export, plus true end-to-end latency.
