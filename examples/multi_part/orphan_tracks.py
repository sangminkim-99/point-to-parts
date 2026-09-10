"""Causal reassociation of visible unowned sparse tracks by rigid motion."""
import numpy as np


class OrphanTracks:
    def __init__(self, min_obs=6, window=20, max_gap=10):
        self.min_obs, self.window, self.max_gap = min_obs, window, max_gap
        self.history = {}
        self.identities = None

    def update(self, frame, points, valid, anchors_valid, parts, excluded=()):
        identities = tuple(p.part_id for p in parts)
        if identities != self.identities:
            self.history.clear()
            self.identities = identities
        owned = set(int(k) for p in parts for k in p.idx) | set(excluded)
        for idx in list(self.history):
            if idx in owned:
                del self.history[idx]
        # A held competing part must not lose simply because its pose is absent.
        if len(parts) < 2 or not all(p.observed for p in parts):
            return []
        accepted = []
        for idx in np.flatnonzero(valid & anchors_valid):
            idx = int(idx)
            if idx in owned:
                continue
            h = self.history.setdefault(idx, [])
            if h and frame <= h[-1][0]:
                continue
            if h and frame - h[-1][0] > self.max_gap:
                h.clear()
            local = np.stack([(points[idx] - p.pose[:3, 3]) @ p.pose[:3, :3] for p in parts])
            if not np.isfinite(local).all():
                continue
            h.append((frame, local))
            del h[:-self.window]
            if len(h) < self.min_obs:
                continue
            samples = np.stack([x for _, x in h])
            means = samples.mean(axis=0)
            spread = np.sqrt(((samples - means) ** 2).sum(axis=2).mean(axis=0))
            order = np.argsort(spread)
            best, runner = int(order[0]), int(order[1])
            noise = max(float(parts[best].sigma), float(parts[runner].sigma), .004)
            # Absolute margin prevents numerical ties / common rigid motion
            # from being mistaken for articulation evidence.
            if spread[best] > 2 * noise or spread[runner] - spread[best] < noise:
                continue
            if np.linalg.norm(local[best] - means[best]) > 2 * noise:
                continue
            accepted.append((idx, best, means[best].copy(), len(h)))
            del self.history[idx]
        return accepted
