# Re-observed sparse track reassociation (experimental)

The user confirmed gray points in the cv2 overlay, not merely missing dense
geometry. The renderer colors tracks gray when they are absent from every
part's index set. Split groups need not cover old invisible/ambiguous tracks;
the existing pending queue principally handles newly seeded tracks.

Opt in with `--set orphan_reassociate=true`. This adds visible, valid, masked,
previously anchored unowned tracks to a separate motion-evidence history. It
skips tracks in the existing pending queue. For each existing observed part,
transform observations into its local coordinates. After at least six distinct
observations, accept only if the best local-coordinate spread is below twice
the noise floor, beats the runner by one noise floor and fits the latest sample.
All competing part poses must be observed. Common rigid motion stays ambiguous.
Part identity-set changes reset histories; long observation gaps restart them.

Accepted sparse tracks receive a new local anchor and begin a new track grace
period. Log: `[orphan] f... track=... -> part=... observations=...`.
This does not restore failed tracker correspondences, discover a completely
unseen face, or repair dense Gaussian labels. Surface ownership and sparse
membership are distinct tasks. Persistent drift or wrong fresh poses can still
misassign a point; no default promotion until matched real replay evaluation.

Validation: five synthetic motion/visibility/pending/duplicate-frame tests;
132 object tests passed. Real box replay and wrong-association assessment are
assigned to Claude Fable and are pending. These tests do not establish real
surface completion or thin-object identity recovery.
