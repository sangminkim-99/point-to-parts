"""Causal pose hypotheses scored against persistent geometry, never GT poses."""
import numpy as np
from examples.multi_part.surface_memory import reprojection_evidence


def choose_pose(points, sparse_pose, predicted_pose, depth, mask, K,
                refine, tolerance=.012, margin=.08, min_support=.3):
    """Rescue a contradicted sparse fit only when refined temporal geometry wins.

    `refine` is a local geometric optimizer supplied by the caller. Both raw and
    refined predictions must compete on the same stored surface sample. This
    does not create pose evidence if the surface is entirely hidden.
    """
    def score(T):
        evidence = reprojection_evidence(points, T, depth, mask, K, tolerance)
        return evidence['support'] - evidence['contradiction'], evidence['support']
    base, _ = score(sparse_pose)
    # Avoid invoking dense optimization for an already well-supported fit.
    if base >= .85:
        return sparse_pose, False
    refined = refine(predicted_pose)
    candidates = [predicted_pose, refined]
    best = sparse_pose
    best_score = base
    rescued = False
    for T in candidates:
        if not np.isfinite(T).all():
            continue
        value, support = score(T)
        if support >= min_support and value > base + margin and value > best_score:
            best, best_score, rescued = T, value, True
    return best, rescued
