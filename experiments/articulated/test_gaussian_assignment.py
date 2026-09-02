"""Does rendering residual separate parts, given correct hypotheses?

Validates the mechanism before it is trusted with RANSAC output: the object is
lifted to Gaussians on the first frame, and each later frame is explained with
the GROUND-TRUTH per-part transforms as hypotheses. If the residual cannot pick
the right part when the hypotheses are exactly right, nothing downstream can.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from point2pose.io.sources.dataset.sapien_reader import open_sequence
from point2pose.pipeline.components.gaussian_part_assignment import (
    GaussianCloud, GaussianPartAssignment,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-dir", required=True)
    ap.add_argument("--anchor", type=int, default=0)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--gauss-stride", type=int, default=2)
    ap.add_argument("--scale-mult", type=float, default=1.2)
    ap.add_argument("--depth-sigma", type=float, default=0.02)
    ap.add_argument("--color-weight", type=float, default=0.3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    r = open_sequence(args.seq_dir)
    parts = r.get_object_names()
    K, H, W = r.K, r.H, r.W
    a = args.anchor

    rgb0, d0 = r.get_color(a), r.get_depth(a)
    obj = np.zeros((H, W), np.uint8)
    for m in r.get_masks(a):
        obj |= m
    cloud = GaussianCloud.from_depth(rgb0, d0, K, obj, stride=args.gauss_stride,
                                     scale_mult=args.scale_mult)
    print(f"[gauss] {len(cloud)} gaussians from frame {a}, parts {parts}")

    # ground-truth label for each Gaussian, for scoring only
    pim = r.render_part_index_map(a)[::args.gauss_stride, ::args.gauss_stride]
    dd = d0[::args.gauss_stride, ::args.gauss_stride]
    mm = obj[::args.gauss_stride, ::args.gauss_stride]
    sel = (dd > 0.05) & (mm > 0)
    gt_lab = pim[sel]

    assign = GaussianPartAssignment(cloud, len(parts), depth_sigma=args.depth_sigma,
                                    color_weight=args.color_weight)

    T_a = {p: r.get_gt_pose(a, p) for p in parts}
    frames = list(range(a, len(r), args.stride))
    rows = []
    for i in frames:
        rgb, dep = r.get_color(i), r.get_depth(i)
        # hypothesis k = "everything moved the way part k moved"
        hyps = [r.get_gt_pose(i, p) @ np.linalg.inv(T_a[p]) for p in parts]
        om = np.zeros((H, W), np.uint8)
        for m in r.get_masks(i):
            om |= m
        st = assign.step(hyps, K, H, W, dep, rgb, obs_mask=om)
        rows.append(st)

    lab = assign.labels()
    ok = lab >= 0
    print(f"[gauss] decided {100*ok.mean():.1f}% of gaussians")
    if ok.sum():
        acc = float((lab[ok] == gt_lab[ok]).mean())
        print(f"[gauss] assignment accuracy on decided gaussians: {100*acc:.1f}%")
        for k, p in enumerate(parts):
            m = lab == k
            if m.sum() == 0:
                print(f"   {p}: none"); continue
            pur = float((gt_lab[m] == k).mean())
            print(f"   {p}: {int(m.sum()):6d} gaussians, purity {100*pur:5.1f}%")
    e = np.array([x["explained_frac"] for x in rows])
    u = np.array([x["unexplained_frac"] for x in rows])
    print(f"[gauss] explained pixels mean {100*e.mean():.1f}%   unexplained {100*u.mean():.1f}%")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        PAL = np.array([[245,135,66],[60,130,214],[100,190,90],[220,100,170]], np.uint8)
        vis = cv2.cvtColor(r.get_color(a), cv2.COLOR_RGB2BGR).copy()
        ys, xs = np.where(sel)
        for (y, x, l) in zip(ys * args.gauss_stride, xs * args.gauss_stride, lab):
            c = PAL[l % len(PAL)] if l >= 0 else np.array([120,120,120], np.uint8)
            cv2.circle(vis, (int(x), int(y)), 1, tuple(int(v) for v in c), -1)
        cv2.imwrite(args.out, vis)
        print(f"[gauss] wrote {args.out}")


if __name__ == "__main__":
    main()
