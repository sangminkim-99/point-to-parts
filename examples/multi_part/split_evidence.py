"""Bounded, part-local geometry retained independently of live map maintenance."""
import numpy as np


class SplitEvidence:
    def __init__(self, max_points=1536):
        self.max_points = max_points
        self.snapshots = {}
        self.topology = None

    def update(self, parts, model, frame):
        topology = tuple(p.part_id for p in parts)
        if topology != self.topology:
            self.snapshots.clear()
            self.topology = topology
        if model is None:
            return
        for j, part in enumerate(parts):
            if part.part_id in self.snapshots or not part.observed or part.over <= 0:
                continue
            idx = np.flatnonzero(model.labels == j)
            idx = idx[::max(1, int(np.ceil(len(idx) / self.max_points)))]
            if not len(idx):
                continue
            points = model.cloud.means[idx].detach().cpu().numpy().copy()
            self.snapshots[part.part_id] = (int(frame), points)
