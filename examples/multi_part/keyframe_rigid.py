"""Small classical RGB-D feature/keyframe baseline, not a BundleTrack reproduction."""
from dataclasses import dataclass
import cv2
import numpy as np


def fit_rigid(source, target):
    a, b = source.mean(0), target.mean(0)
    U, _, Vt = np.linalg.svd((source - a).T @ (target - b))
    R = Vt.T @ np.diag([1., 1., np.linalg.det(Vt.T @ U.T)]) @ U.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, b - R @ a
    return T


def robust_rigid(source, target, threshold=.008, trials=128, min_inliers=8):
    if len(source) < min_inliers:
        return None
    rng = np.random.default_rng(0)
    best = np.zeros(len(source), bool)
    for _ in range(trials):
        idx = rng.choice(len(source), 3, replace=False)
        if np.linalg.norm(np.cross(source[idx[1]]-source[idx[0]], source[idx[2]]-source[idx[0]])) < 1e-7:
            continue
        T = fit_rigid(source[idx], target[idx])
        err = np.linalg.norm(source @ T[:3, :3].T + T[:3, 3] - target, axis=1)
        inside = err < threshold
        if inside.sum() > best.sum():
            best = inside
    if best.sum() < min_inliers:
        return None
    T = fit_rigid(source[best], target[best])
    error = np.linalg.norm(source @ T[:3, :3].T + T[:3, 3] - target, axis=1)
    keep = error < threshold
    if keep.sum() < min_inliers:
        return None
    return T, keep, float(np.median(error[keep]))


@dataclass
class Keyframe:
    frame: int
    xyz: np.ndarray
    descriptors: object
    pose: np.ndarray


class RGBDKeyframes:
    def __init__(self, K, memory=True, max_keyframes=12, interval=10, backend="sift", multi_reference=False):
        self.K = K
        self.memory, self.limit, self.interval = memory, max_keyframes, interval
        self.multi_reference = multi_reference
        self.backend = backend
        if backend == 'sift':
            self.detector = cv2.SIFT_create(nfeatures=1600)
            self.matcher = cv2.BFMatcher()
        elif backend == 'lightglue':
            from lightglue import SuperPoint, LightGlue
            self.detector = SuperPoint(max_num_keypoints=1024).eval().cuda()
            self.matcher = LightGlue(features='superpoint').eval().cuda()
        else:
            raise ValueError(f'unknown feature backend: {backend}')
        self.keyframes = []
        self.previous = None
        self.pose = np.eye(4)

    def extract(self, rgb, depth, mask):
        if self.backend == 'lightglue':
            import torch
            image = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().cuda() / 255
            with torch.inference_mode():
                features = self.detector.extract(image, resize=None)
            xy = features['keypoints'][0].cpu().numpy()
            xyz, good = self.lift(xy, depth, mask)
            for name in ('keypoints', 'descriptors', 'keypoint_scores'):
                if name in features:
                    features[name] = features[name][:, torch.as_tensor(good, device='cuda')]
            return xyz, features
        keypoints, descriptors = self.detector.detectAndCompute(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (mask > 0).astype(np.uint8)*255)
        if descriptors is None:
            return np.empty((0, 3)), np.empty((0, 128), np.float32)
        xy = np.array([p.pt for p in keypoints])
        xyz, good = self.lift(xy, depth, mask)
        # RootSIFT descriptor normalization, a classical baseline component.
        descriptors = np.sqrt(descriptors / np.maximum(descriptors.sum(1, keepdims=True), 1e-8))
        return xyz, descriptors[good]

    def lift(self, xy, depth, mask):
        u = np.clip(np.rint(xy[:, 0]).astype(int), 0, depth.shape[1]-1)
        v = np.clip(np.rint(xy[:, 1]).astype(int), 0, depth.shape[0]-1)
        z = depth[v, u]
        good = np.isfinite(z) & (z > .05) & (mask[v, u] > 0)
        xyz = np.column_stack([(xy[:, 0]-self.K[0, 2])*z/self.K[0, 0],
                               (xy[:, 1]-self.K[1, 2])*z/self.K[1, 1], z])
        return xyz[good], good

    def matches(self, first, second):
        if self.backend == 'lightglue':
            import torch
            with torch.inference_mode():
                matches = self.matcher({'image0': first, 'image1': second})['matches'][0]
            return matches.cpu().numpy()
        pairs = self.matcher.knnMatch(first, second, k=2)
        reverse = self.matcher.match(second, first)
        back = {m.queryIdx: m.trainIdx for m in reverse}
        return np.array([(a.queryIdx, a.trainIdx) for row in pairs if len(row) == 2 for a, b in [row]
                         if a.distance < .75*b.distance and back.get(a.trainIdx) == a.queryIdx], dtype=int).reshape(-1, 2)

    def step(self, frame, rgb, depth, mask):
        xyz, desc = self.extract(rgb, depth, mask)
        if self.previous is None:
            if len(xyz) < 8:
                raise ValueError("at least eight valid RGB-D features required for initialization")
            first = Keyframe(frame, xyz, desc, self.pose.copy())
            self.previous = first
            self.keyframes.append(first)
            return self.pose.copy(), True, len(xyz), frame
        candidates = [self.previous] + (self.keyframes if self.memory else [])
        best = None
        constraints = []
        visited = set()
        for k in candidates:
            if k.frame in visited or min(len(xyz), len(k.xyz)) < 8:
                continue
            visited.add(k.frame)
            matches = self.matches(k.descriptors, desc)
            if len(matches) < 8:
                continue
            source = k.xyz[matches[:, 0]]
            target = xyz[matches[:, 1]]
            fit = robust_rigid(source, target)
            if fit is None:
                continue
            T, keep, error = fit
            rank = (int(keep.sum()), -error)
            anchor = (source[keep] - k.pose[:3, 3]) @ k.pose[:3, :3]
            constraints.append((rank, anchor, target[keep]))
            if best is None or rank > best[0]:
                best = rank, T @ k.pose, k.frame
        if best is None:
            # Explicit failure; do not grow memory from a held/unobserved pose.
            return self.pose.copy(), False, 0, -1
        rank, self.pose, reference = best
        if self.multi_reference and len(constraints) >= 2:
            selected = sorted(constraints, key=lambda row: row[0], reverse=True)[:4]
            source, target = [], []
            for _, a, b in selected:
                idx = np.linspace(0, len(a)-1, min(32, len(a))).astype(int)
                source.append(a[idx]); target.append(b[idx])
            pooled = robust_rigid(np.concatenate(source), np.concatenate(target))
            if pooled is not None:
                self.pose = pooled[0]
        current = Keyframe(frame, xyz, desc, self.pose.copy())
        self.previous = current
        if self.memory and frame - self.keyframes[-1].frame >= self.interval:
            self.keyframes.append(current)
            if len(self.keyframes) > self.limit:
                self.keyframes.pop(1)  # preserve the original reference for revisits
        return self.pose.copy(), True, rank[0], reference
