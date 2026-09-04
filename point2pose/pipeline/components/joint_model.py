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

    def __init__(self, min_obs=8, min_angle_deg=1.5, min_shift=0.004):
        # Refusing to fit below 6 degrees or 2 cm hides exactly the case the
        # wiggle demo is about. The fit is attempted far earlier and confidence()
        # reports how little the joint has been exercised instead.
        self.A = []
        self.min_obs = int(min_obs)
        self.min_angle = np.radians(min_angle_deg)
        self.min_shift = float(min_shift)
        self.kind = None          # "revolute" | "prismatic"
        self.axis = None          # unit direction, parent frame
        self.point = None         # a point on the axis (revolute only)
        self.A0 = None            # the relative transform at value 0
        self.conf = None          # dict from confidence(); None until fitted
        self.sigma = 0.003        # metres of observation noise, for the BIC
        self.sigma_eff = 0.003    # scale actually used, from the best fit
        self.bic = {}             # per-candidate BIC, lowest wins

    def add(self, A):
        # a degenerate rotation block gives scipy a zero-norm quaternion
        if not np.all(np.isfinite(A)) or abs(np.linalg.det(A[:3, :3]) - 1) > 0.2:
            return
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
        """Axis by least squares through the observed translations."""
        nrm = np.linalg.norm(t, axis=1)
        if nrm.max() < self.min_shift:
            return None
        c = t - t.mean(0)
        # the principal direction is the axis; one extreme sample is not enough
        u, sv, vt = np.linalg.svd(c, full_matrices=False)
        d = vt[0]
        if float(d @ t[np.argmax(nrm)]) < 0:
            d = -d
        return "prismatic", d, np.zeros(3)

    # Sturm, Stachniss & Burgard (JAIR 2011) count k = 6 for a rigid link,
    # 9 for prismatic and 12 for revolute, and select on the lowest BIC.
    # "disconnected" is free 6-DoF motion: it explains anything, and pays 6
    # parameters per observation for it, the way Sturm's non-parametric model
    # does. Without it in the set a spurious part must be typed revolute or
    # prismatic, because those are the only answers on offer.
    K_PARAMS = {"rigid": 6, "prismatic": 9, "revolute": 12}
    K_FREE_PER_OBS = 6

    def _score(self, model, A):
        """Mean squared residual of a candidate over the offset-free history."""
        kind, axis, point = model
        keep = (self.kind, self.axis, self.point, self.A0)
        self.kind, self.axis, self.point, self.A0 = kind, axis, point, None
        r = float(np.mean([self.residual(a) ** 2 for a in A]))
        self.kind, self.axis, self.point, self.A0 = keep
        return r

    @staticmethod
    def _noise_from_smoothness(A):
        """Noise scale from the high-frequency content of the pose sequence.

        Real relative motion is smooth in time and noise is not, so the second
        difference is almost all noise. Independent of every candidate model,
        which is what makes the comparison between them mean anything.
        """
        if len(A) < 5:
            return 0.0
        t = A[:, :3, 3]
        d2 = t[2:] - 2 * t[1:-1] + t[:-2]
        m = np.linalg.norm(d2, axis=1)
        if not m.size:
            return 0.0
        # var of a second difference of white noise is 6 sigma^2
        return float(1.4826 * np.median(np.abs(m - np.median(m)))
                     + np.median(m)) / np.sqrt(6.0)

    def _bic_free(self, n):
        """A model that fits everything and pays for every degree of freedom."""
        return self.K_FREE_PER_OBS * n * np.log(max(n, 2))

    def _bic(self, kind, mse, n, sigma=None):
        """BIC(M) = -2 log p + k log n, Gaussian noise, constants dropped.

        A pure translation is explained perfectly by a revolute joint with a
        very large radius -- Sturm et al. measure 1.7 mm for the drawer against
        1.6 mm for the prismatic fit -- so residual alone can never separate
        them. Revolute pays for its three extra parameters instead.
        """
        s2 = max(self.sigma if sigma is None else sigma, 1e-4) ** 2
        return n * mse / s2 + self.K_PARAMS[kind] * np.log(max(n, 2))

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
        raw = np.stack(self.A)
        # Both models describe motion FROM a rest pose, but two parts each carry
        # their own anchor frame, so the relative transform at rest is never the
        # identity. Without factoring that offset out, the fixed rotation between
        # the parts is charged to the prismatic model as permanent residual and a
        # sliding joint is always typed revolute. This is Sturm's origin a.
        self.A0 = raw[0].copy()
        A0i = np.linalg.inv(self.A0)
        A = np.einsum("ij,njk->nik", A0i, raw)
        ang = np.linalg.norm(R.from_matrix(A[:, :3, :3]).as_rotvec(), axis=1)
        t = A[:, :3, 3]
        # Sturm's model set includes a rigid link, and it is what says "these
        # two are one body": unstructured relative motion from a noisy pose on a
        # handful of points fits no 1-DoF manifold, so rigid wins on BIC even
        # though the parts appear to move.
        cands = [("rigid", np.array([0., 0., 1.]), np.zeros(3))]
        if ang.max() >= self.min_angle:
            m = self._fit_revolute(A, ang, t)
            if m is not None:
                cands.append(m)
        m = self._fit_prismatic(A, ang, t)
        if m is not None:
            cands.append(m)
        if not cands:
            return False
        n = len(A)
        mses = {m[0]: self._score(m, A) for m in cands}
        # The observations are relative POSES from a RANSAC fit on a handful of
        # points, not points, so the per-point noise underestimates them by
        # orders of magnitude and no rigid model could ever compete. The scale
        # must not come from the model being judged either -- taking it from the
        # best 1-DoF fit makes that fit adequate by construction, and nothing
        # can ever beat it. It comes from the data alone instead.
        self.sigma_eff = max(self.sigma, self._noise_from_smoothness(A))
        rows = [(self._bic(m[0], mses[m[0]], n, self.sigma_eff), mses[m[0]], m)
                for m in cands]
        rows.append((self._bic_free(n), 0.0,
                     ("disconnected", np.array([0., 0., 1.]), np.zeros(3))))
        rows.sort(key=lambda x: x[0])
        self.bic = {m[0]: b for b, _, m in rows}
        self.kind, self.axis, self.point = rows[0][2]
        self._confidence(A, ang, t, rows)
        return True

    # --------------------------------------------------------- confidence --
    def _confidence(self, A, ang, t, rows, boots=16,
                    axis_scale_deg=8.0, min_range_deg=8.0, min_range_m=0.02):
        """How sure the fit is, split into the three things that can be wrong.

        A joint can be mis-typed, its axis can be loose, or it can simply not
        have been exercised: 2 degrees of observed motion pins no axis however
        small the residual. Reporting one number without the three would hide
        which of them the human has to fix by moving the object more.
        """
        rng = np.random.default_rng(0)
        # type: a BIC difference is approximately twice a log Bayes factor
        b = np.array([x[0] for x in rows], dtype=np.float64)
        w = np.exp(-0.5 * np.clip(b - b.min(), 0, 700))
        type_p = float(w[0] / w.sum()) if w.size > 1 else 1.0

        # axis: the spread of the axis over bootstrap resamples of the history
        keep = (self.kind, self.axis, self.point)
        axes = []
        n = len(A)
        for _ in range(boots if n >= 6 else 0):
            i = rng.integers(0, n, n)
            m = (self._fit_revolute(A[i], ang[i], t[i]) if self.kind == "revolute"
                 else self._fit_prismatic(A[i], ang[i], t[i]))
            if m is not None:
                axes.append(m[1] * np.sign(np.dot(m[1], keep[1]) or 1.0))
        self.kind, self.axis, self.point = keep
        if len(axes) >= 4:
            V = np.stack(axes)
            mean = V.mean(0)
            mean /= np.linalg.norm(mean) + 1e-12
            axis_std = float(np.degrees(np.arccos(
                np.clip(V @ mean, -1, 1)).std()))
        else:
            axis_std = float("nan")

        # excitation: how far the joint has actually been moved (A is offset-free)
        a0, self.A0 = self.A0, None
        vals = np.array([self.value_of(a) for a in A])
        self.A0 = a0

        # Smoothness of q(t). Residual-based evidence cannot separate a joint
        # from correlated tracking error -- a 1-DoF model fits any jitter better
        # than a rigid one, because random 3D points project onto their
        # principal axis. But a real joint is DRIVEN, so q(t) is smooth, while
        # error gives a q(t) that is white. Independent of every residual.
        smooth = 0.0
        if vals.size >= 5:
            d2 = vals[2:] - 2 * vals[1:-1] + vals[:-2]
            v_all = float(np.var(vals))
            v_hf = float(np.var(d2)) / 6.0     # white noise inflates by 6
            smooth = float(np.clip(1.0 - v_hf / max(v_all, 1e-12), 0.0, 1.0))
        span = float(vals.max() - vals.min()) if vals.size else 0.0
        need = np.radians(min_range_deg) if self.kind == "revolute" else min_range_m
        exc = float(np.clip(span / max(need, 1e-9), 0.0, 1.0))

        axis_ok = 1.0 if axis_std != axis_std else \
            float(np.exp(-axis_std / axis_scale_deg))
        self.conf = {"type_p": type_p, "axis_std_deg": axis_std, "span": span,
                     "excitation": exc, "smooth": smooth, "n": int(n),
                     "rmse": float(np.sqrt(rows[0][1])),
                     "bic": dict(self.bic),
                     "conf": float(type_p * axis_ok * exc * smooth)}
        return self.conf

    def confidence(self):
        """The last fit's confidence, or zeros if it has never been fitted."""
        return self.conf or {"type_p": 0.0, "axis_std_deg": float("nan"),
                             "span": 0.0, "excitation": 0.0, "smooth": 0.0,
                             "n": len(self.A),
                             "rmse": float("nan"), "bic": {}, "conf": 0.0}

    # --------------------------------------------------------------- pose --
    def at(self, value):
        """Relative transform at a joint value (radians, or metres)."""
        return (self.A0 if self.A0 is not None else np.eye(4)) @ self._local(value)

    def _local(self, value):
        """Joint motion away from the rest pose."""
        A = np.eye(4)
        if self.kind in ("rigid", "disconnected"):
            return A
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
        if self.A0 is not None:
            A = np.linalg.inv(self.A0) @ A
        if self.kind == "revolute":
            rv = R.from_matrix(A[:3, :3]).as_rotvec()
            return float(rv @ self.axis)
        if self.kind == "prismatic":
            return float(A[:3, 3] @ self.axis)
        return 0.0        # rigid and unfitted both have no configuration

    def residual(self, A):
        """How far a relative transform is from the joint manifold, in metres."""
        if self.kind is None:
            return float("inf")
        B = self.at(self.value_of(A))
        E = np.linalg.inv(B) @ np.asarray(A, dtype=np.float64)
        rot = np.linalg.norm(R.from_matrix(E[:3, :3]).as_rotvec())
        return float(np.linalg.norm(E[:3, 3]) + 0.1 * rot)
