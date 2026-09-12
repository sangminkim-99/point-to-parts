#!/usr/bin/env python
"""Purity of split partitions at birth, raw vs residual-refined (offline).

Reads `replay.py --dump-split-diag` output. For every accepted split it
reports, per child: the GT-label composition of the group as installed
(`raw`), the fraction of its points that fit the OTHER child's motion better
(the mixture signature, GT-free), and what the composition WOULD be if each
point were handed to the motion with the smaller residual, dropping points
in the ambiguous band (`refined`) -- the same rule the single-frame split
path already applies and the co-association path does not.

GT labels are per track (where the track was born), evaluation only.
"""
import argparse
import json

import numpy as np


def composition(idx, lab, names):
    l = np.array([lab[i] if i < len(lab) else -1 for i in idx])
    l = l[l >= 0]
    if l.size == 0:
        return {}, 0.0
    c = {names[k]: int((l == k).sum()) for k in np.unique(l)}
    return c, float(max(c.values()) / l.size)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("diag")
    ap.add_argument("--band", type=float, default=None,
                    help="ambiguous band (x own residual) a point must clear to move; "
                         "default: the run's ambiguous_band")
    args = ap.parse_args()
    d = json.load(open(args.diag))
    lab, names = d["gt_lab"], d["gt_parts"]
    band = args.band if args.band is not None else d["ambiguous_band"]
    thr = 1000 * d["inlier_thres"]
    print(f"{len(d['splits'])} accepted splits; inlier {thr:.0f} mm, band x{band}")
    for sp in d["splits"]:
        if "children" not in sp:
            continue
        ch = sp["children"]
        print(f"\n== f{sp['frame']} via {sp['via']} parent p{sp['parent_part_id']} sigma {sp['sigma_mm']:.1f} mm")
        # refined partition: every point of both children goes to the motion it fits better
        all_idx = [i for c in ch for i in c["idx"]]
        own = [np.array(c["own_mm"]) for c in ch]
        oth = [np.array(c["oth_mm"]) for c in ch]
        # residual under motion 0 / motion 1 for each point
        r0 = np.concatenate([own[0], oth[1]])
        r1 = np.concatenate([oth[0], own[1]])
        best = np.where(r0 <= r1, 0, 1)
        bv, sv = np.minimum(r0, r1), np.maximum(r0, r1)
        keep = (bv < thr) & (sv > band * bv)
        for k, c in enumerate(ch):
            raw, pur = composition(c["idx"], lab, names)
            ref_idx = [i for i, b, kp in zip(all_idx, best, keep) if b == k and kp]
            rc, rp = composition(ref_idx, lab, names)
            print(f"  child{k}: n={c['n']:>3} raw {raw} purity {pur:.2f} | own med {c['own_med_mm']:.1f} mm, "
                  f"other med {c['oth_med_mm']:.1f} mm, own inlier {c['own_inlier_frac']:.2f}, "
                  f"prefers other {c['prefers_other_frac']:.2f} (clear {c['prefers_other_clear_frac']:.2f})"
                  f"\n          refined n={len(ref_idx):>3} {rc} purity {rp:.2f}")


if __name__ == "__main__":
    main()
