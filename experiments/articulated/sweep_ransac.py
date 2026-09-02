"""Sweep sequential-RANSAC parameters over cached tracks.

Answers the question the part-spawn rule depends on: is a secondary consensus
set temporally persistent, and does it correspond to a real moving part?
Reuses tracks cached by demo_part_discovery.py --save-tracks so the tracker does
not have to re-run for every parameter setting.
"""

import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from point2pose.modules.register.svd_cluster_ransac_register import (
    SVDClusterRANSACRegister,
)
from experiments.articulated.demo_part_discovery import sequential_ransac


def evaluate(d, inlier_thres, min_inliers, ransac_iters, max_clusters, seed=0):
    np.random.seed(seed)
    reg = SVDClusterRANSACRegister({
        "ransac_iters": ransac_iters, "sample_size": 4,
        "inlier_thres": inlier_thres, "min_inliers": min_inliers,
        "max_clusters": max_clusters, "use_uncertainty": False,
    })
    anchor_xyz, anchor_valid = d["anchor_xyz"], d["anchor_valid"]
    gt_label, gt_parts = d["gt_label"], [str(p) for p in d["gt_parts"]]

    prev_sec = []
    sec_jac, sec_pur, sec_n, prim_jac, prim_pur, ncl = [], [], [], [], [], []
    prev_prim = set()

    for k in range(len(d["xyz"])):
        usable = np.where(anchor_valid & d["valid"][k] & d["visible"][k])[0]
        if len(usable) < max(min_inliers, 8):
            continue
        cl = sequential_ransac(reg, anchor_xyz[usable], d["xyz"][k][usable])
        for c in cl:
            c["inliers"] = usable[c["inliers"]]
        cl.sort(key=lambda c: -c["ninliers"])
        if not cl:
            continue
        ncl.append(len(cl))

        def purity(idx):
            lab = gt_label[idx]
            lab = lab[lab >= 0]
            if lab.size == 0:
                return 0.0
            return np.bincount(lab).max() / lab.size

        prim = set(cl[0]["inliers"].tolist())
        if prev_prim:
            prim_jac.append(len(prim & prev_prim) / len(prim | prev_prim))
        prim_pur.append(purity(cl[0]["inliers"]))
        prev_prim = prim

        sec = [set(c["inliers"].tolist()) for c in cl[1:]]
        for s_, c in zip(sec, cl[1:]):
            if prev_sec:
                sec_jac.append(max(len(s_ & p) / len(s_ | p) for p in prev_sec))
            sec_pur.append(purity(c["inliers"]))
            sec_n.append(c["ninliers"])
        prev_sec = sec

    f = lambda a: float(np.mean(a)) if len(a) else float("nan")
    return {
        "n_clusters": f(ncl),
        "primary_jaccard": f(prim_jac), "primary_purity": f(prim_pur),
        "secondary_jaccard": f(sec_jac), "secondary_purity": f(sec_pur),
        "secondary_size": f(sec_n),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--inlier-thres", type=float, nargs="+",
                    default=[0.004, 0.008, 0.012, 0.02, 0.03, 0.05])
    ap.add_argument("--min-inliers", type=int, nargs="+", default=[5])
    ap.add_argument("--ransac-iters", type=int, nargs="+", default=[100])
    ap.add_argument("--max-clusters", type=int, default=4)
    args = ap.parse_args()

    d = dict(np.load(args.tracks, allow_pickle=True))
    print(f"tracks: {d['xyz'].shape[0]} frames x {d['xyz'].shape[1]} points  "
          f"GT parts {[str(p) for p in d['gt_parts']]}\n")
    hdr = (f"{'thres':>7s} {'min_inl':>8s} {'iters':>6s} | {'clusters':>8s} "
           f"{'prim_J':>7s} {'prim_pur':>9s} | {'sec_J':>6s} {'sec_pur':>8s} {'sec_n':>6s}")
    print(hdr); print("-" * len(hdr))
    for th, mi, it in itertools.product(args.inlier_thres, args.min_inliers,
                                        args.ransac_iters):
        r = evaluate(d, th, mi, it, args.max_clusters)
        print(f"{th*1000:6.0f}mm {mi:8d} {it:6d} | {r['n_clusters']:8.2f} "
              f"{r['primary_jaccard']:7.3f} {r['primary_purity']:9.3f} | "
              f"{r['secondary_jaccard']:6.3f} {r['secondary_purity']:8.3f} "
              f"{r['secondary_size']:6.1f}")


if __name__ == "__main__":
    main()
