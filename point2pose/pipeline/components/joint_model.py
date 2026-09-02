"""One-DoF joint model fitted online from a part's tracked poses.

Dense per-part tracking is a free 6-DoF search, and that is what makes it fail
on exactly the parts articulation is about. Measured on the simulated laptop:
the lid is segmented correctly (94% purity) but swings 132 degrees, its own
sparse tracks vanish as the face it sits on turns away from the camera, and the
pose search then wanders off by 800 mm. The energy landscape is not the problem
-- the ground-truth pose scores four times better than the one chosen -- the
search is.

An articulated part does not have six degrees of freedom relative to its parent.
It has one. Once the axis is known, the pose is a scalar, and a scalar can be
searched exhaustively every frame, so the estimate cannot drift away and cannot
be recovered only by luck.

The joint is fitted from the poses the tracker already trusts, so nothing is
assumed about the object in advance -- neither the axis, nor its type, nor how
many joints there are. This is the plan's section 4 (joint estimation) brought
forward, and the screw parameterisation is standard rigid-body kinematics; what
is specific here is using it as a *search constraint* during causal tracking
rather than as an offline output.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class JointModel:
    """Revolute or prismatic joint between a part and its parent.

    Feed it relative transforms `A = inv(T_parent) @ T_part`; once enough of
    them span a real range of motion it fits an axis and can then generate a
    pose for any joint value.
    """

    def __init__(self, min_obs=8, min_angle_deg=6.0, min_shift=0.02):
        self.A = []
        self.min_obs = int(min_obs)
        self.min_angle = np.radians(min_angle_deg)
        self.min_shift = float(min_shift)
        self.kind = None          # "revolute" | "prismatic"
        self.axis = None          # unit direction, parent frame
        self.point = None         # a point on the axis (revolute only)
        self.A0 = None            # the relative transform at value 0

    def add(self, A):
        self.A.append(np.asarray(A, dtype=np.float64).copy())

    # ---------------------------------------------------------------- fit --
    def _fit_revolute(self, A, ang, t):
        big = ang > self.min_angle * 0.3
        if big.sum() < 3:
            return None
        rv = R.from_matrix(A[big, :3, :3]).as_rotvec()
        u = rv / ang[big][:, None]
        # sign-align before averaging: the axis is a line, not an arrow, and a
        # part that swings both ways would otherwise average to nothing
        ref = u[np.argmax(ang[big])]
        n = (u * np.sign(u @ ref)[:, None]).mean(axis=0)
        n /= np.linalg.norm(n) + 1e-12
        # Every observation must satisfy (I - R_i) p = t_i for the same p.
        # Weighted by rotation angle: (I - R) is near-singular for a small
        # rotation, so a part that has only just started to swing contributes an
        # almost unconstrained row. The laptop lid's joint is first fitted from
        # under 25 degrees of motion, and an unweighted solve put its axis about
        # 110 mm off -- enough that the joint sweep bracketed the true pose
        # without ever landing on it.
        idx = np.where(big)[0]
        w = ang[idx]
        M = np.concatenate([(np.eye(3) - A[i, :3, :3]) * wi
                            for i, wi in zip(idx, w)], 0)
        b = np.concatenate([t[i] * wi for i, wi in zip(idx, w)], 0)
        p, *_ = np.linalg.lstsq(M, b, rcond=None)
        return "revolute", n, p

    def _fit_prismatic(self, A, ang, t):
        nrm = np.linalg.norm(t, axis=1)
        if nrm.max() < self.min_shift:
            return None
        d = t[np.argmax(nrm)]
        return "prismatic", d / (np.linalg.norm(d) + 1e-12), np.zeros(3)

    def _score(self, model, A):
        kind, axis, point = model
        keep = (self.kind, self.axis, self.point)
        self.kind, self.axis, self.point = kind, axis, point
        r = float(np.mean([self.residual(a) for a in A]))
        self.kind, self.axis, self.point = keep
        return r

    def fit(self):
        """Fit the axis from the accumulated relative transforms.

        Both joint types are fitted and the one that explains the history better
        wins, and the whole fit is redone as observations accumulate. Deciding
        the type once, early, gets it wrong: the simulated laptop lid's first
        trusted frames carry under 6 degrees of rotation, so a fit-once model
        called it prismatic and never reconsidered.
        """
        if len(self.A) < self.min_obs:
            return False
        A = np.stack(self.A)
        ang = np.linalg.norm(R.from_matrix(A[:, :3, :3]).as_rotvec(), axis=1)
        t = A[:, :3, 3]
        cands = []
        if ang.max() >= self.min_angle:
            m = self._fit_revolute(A, ang, t)
            if m is not None:
                cands.append(m)
        m = self._fit_prismatic(A, ang, t)
        if m is not None:
            cands.append(m)
        if not cands:
            return False
        best = min(cands, key=lambda m: self._score(m, A))
        self.kind, self.axis, self.point = best
        return True

    # --------------------------------------------------------------- pose --
    def at(self, value):
        """Relative transform at a joint value (radians, or metres)."""
        A = np.eye(4)
        if self.kind == "revolute":
            Rm = R.from_rotvec(self.axis * value).as_matrix()
            A[:3, :3] = Rm
            A[:3, 3] = self.point - Rm @ self.point
        elif self.kind == "prismatic":
            A[:3, 3] = self.axis * value
        return A

    def value_of(self, A):
        """Joint value that best explains a relative transform."""
        A = np.asarray(A, dtype=np.float64)
        if self.kind == "revolute":
            rv = R.from_matrix(A[:3, :3]).as_rotvec()
            return float(rv @ self.axis)
        if self.kind == "prismatic":
            return float(A[:3, 3] @ self.axis)
        return 0.0

    def residual(self, A):
        """How far a relative transform is from the joint manifold, in metres."""
        if self.kind is None:
            return float("inf")
        B = self.at(self.value_of(A))
        E = np.linalg.inv(B) @ np.asarray(A, dtype=np.float64)
        rot = np.linalg.norm(R.from_matrix(E[:3, :3]).as_rotvec())
        return float(np.linalg.norm(E[:3, 3]) + 0.1 * rot)
