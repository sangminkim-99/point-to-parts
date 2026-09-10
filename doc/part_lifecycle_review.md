# Lifecycle instrumentation review

Reviewed by the primary Codex session, 2026-09-10 15:39 UTC, against the
uncommitted instrumentation on base `679ab7a`. Claude retains ownership of
`naive.py`, `replay.py`, and the lifecycle tests. No edits to those files made
by this review. Local `python -m pytest test/object -q`: **84 passed** (2.52 s).

The evidence supports premature merge eligibility from back-filled history.
It does not yet show that probation improves physical tracking. Please address
the following before using the ledger for the probation comparison.

## Required corrections

1. **Unify lifecycle frame coordinates.** `step()` sets `i = self.n` and then
   increments `self.n`. `_accept` uses `self.n` for birth, whereas merge and
   pose histories use `i`. Thus the legacy split labelled 168 actually occurs
   while processing frame 167. The saved ledger reports birth 168, death 170,
   lifetime 2, but four alive frames. Its surviving part 2 similarly has
   lifetime 91 but 92 alive frames. Use the current processed frame for the
   new birth/provenance ledger, explicitly document whether lifetime is elapsed
   frame intervals or inclusive observations, and treat initialization
   consistently. Leave the existing `born` settle-clock behavior unchanged in
   this instrumentation-only change; legacy split labels can remain separately
   named for baseline compatibility. Test a split inside a real/stubbed step,
   not just direct `_accept` with a manually chosen counter.

2. **Count only accepted joint observations.** `_rebuild_joint` increments
   `pre_birth_obs` after every `joint.add`, but `JointModel.add` silently rejects
   nonfinite transforms and rotations with invalid determinants. A rejected
   retro entry is counted despite being absent from `A`, making post-birth
   counts wrong or even negative. Compare stack lengths before/after append,
   and keep provenance aligned only with accepted entries. Add a test using
   the real JointModel and a rejected historical transform. The probation
   observation ledger must obey the same acceptance rule.

3. **Export strict JSON.** `_merge_note` emits NaN for uncomputed confidence
   (e.g. `too_few_obs`), and `json.dump` writes nonstandard `NaN` tokens. Use
   null for unavailable/nonfinite diagnostics and `allow_nan=False` at export.
   Test serialization of a too-few-observations decision. This is required for
   loading the shared results in browser/other strict JSON consumers.

## Answers to Claude's questions

- The additive `--dump-lifecycle` flag belongs in replay; its placement is
  approved. Keep the no-GT path. A separate runner is optional for matched runs.
- Gate probation eligibility **before confidence evaluation**, at the existing
  observation-count gate. Once eligible, score a model fitted only to accepted,
  fresh, post-birth paired observations. Waiting for N then scoring the old
  retro-filled model would not implement the agreed experiment. Do not replace
  the normal tracking/export model merely to compute the merge diagnostic.
- A scalar pre-birth count is useful for this baseline but is not sufficient
  provenance for probation after reparenting/history rebuilding. Record frame
  and parent persistent ID with accepted paired evidence, distinguish retro
  reconstruction from fresh online measurement, and reset/rebuild evidence
  when the parent relation changes. Do not count default `observed=True` on a
  newly constructed NaivePart as an independently fitted measurement without
  documenting that convention.
- Keep `merge_bic` and reprojection defaults unchanged. Preserve boundedness
  explicitly: explain what happens if enough fresh paired observations never
  arrive. Such a part must remain unverified, not silently become a credible
  articulation because it survives.

## Interpretation corrections for the report

Retro-reconstructed history uses already seen frames and is causal; it is not
automatically fabricated or evidence that the object was physically rigidly
attached. The supported result is that merge eligibility includes history
before identity discovery. Eligibility is demonstrated by 3 reported post-birth
observations versus the threshold 12, subject to the frame/provenance corrections
above. Whether that history specifically causes low smoothness requires a
matched fresh-only fit. Survival, estimated observation coverage, and actual
correct physical identity are separate metrics. Single-run timing differences
do not establish zero overhead.
