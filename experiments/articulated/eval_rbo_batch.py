"""Run part discovery over several RBO sequences and aggregate the scores.

Tuning on a single clip is how a method gets fitted to one sequence, so this
runs a whole set and reports the spread. Each sequence runs in its own process
so a failure in one does not take the batch down.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def run_one(seq_dir, out_dir, extra):
    name = os.path.basename(seq_dir.rstrip("/"))
    out = os.path.join(out_dir, f"{name}.mp4")
    rep = os.path.splitext(out)[0] + "_report.json"
    if os.path.exists(rep):
        return json.load(open(rep)), "cached"
    cmd = [sys.executable, os.path.join(HERE, "run_rbo_part_discovery.py"),
           "--seq-dir", seq_dir, "--out", out,
           "--work-dir", os.path.join(out_dir, "work")] + extra
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if not os.path.exists(rep):
        tail = "\n".join(r.stdout.strip().splitlines()[-4:])
        err = "\n".join(r.stderr.strip().splitlines()[-4:])
        return None, f"FAILED\n{tail}\n{err}"
    return json.load(open(rep)), "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--object", required=True, help="e.g. cabinet, pliers, ikeasmall")
    ap.add_argument("--sequences-root",
                    default="/home/smkim/workspace/dataset/RBO/sequences")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--out-dir", default="debug/multi-parts/batch")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="everything after -- is passed to the runner")
    args = ap.parse_args()

    extra = [a for a in args.rest if a != "--"]
    seqs = sorted(glob.glob(os.path.join(args.sequences_root, f"{args.object}*_o")))
    seqs = [s for s in seqs if re.match(rf"^{args.object}\d+_o$", os.path.basename(s))]
    seqs = seqs[: args.limit]
    if not seqs:
        raise SystemExit(f"no prepared sequences for {args.object}")

    out_dir = os.path.join(args.out_dir, args.object)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[batch] {args.object}: {len(seqs)} sequences, extra args {extra}\n")

    rows = []
    for s in seqs:
        rep, status = run_one(s, out_dir, extra)
        name = os.path.basename(s)
        if rep is None:
            print(f"  {name:18s} {status}")
            continue
        n_gt = len(rep["gt_parts"])
        rows.append({
            "seq": name, "n_obj": rep["n_objects"], "n_gt": n_gt,
            "covered": len(rep.get("gt_covered", [])),
            "purity": rep.get("mean_purity", 0.0),
            "spawns": len(rep.get("spawns", [])),
        })
        print(f"  {name:18s} objects {rep['n_objects']:2d}/{n_gt}  "
              f"covered {rows[-1]['covered']}/{n_gt}  "
              f"purity {rows[-1]['purity']*100:5.1f}%  ({status})")

    if not rows:
        raise SystemExit("no sequence produced a report")

    print(f"\n[batch] === {args.object}: {len(rows)} sequences ===")
    f = lambda k: np.array([r[k] for r in rows], dtype=float)
    n_gt = rows[0]["n_gt"]
    print(f"  objects found : mean {f('n_obj').mean():.1f}  "
          f"range {int(f('n_obj').min())}-{int(f('n_obj').max())}  (GT {n_gt})")
    print(f"  GT covered    : mean {f('covered').mean():.2f}/{n_gt}  "
          f"({100*f('covered').mean()/n_gt:.0f}%)")
    print(f"  mean purity   : {f('purity').mean()*100:.1f}%  "
          f"+/- {f('purity').std()*100:.1f}  "
          f"range {f('purity').min()*100:.0f}-{f('purity').max()*100:.0f}%")
    print(f"  over-seg      : mean {(f('n_obj') - n_gt).mean():+.1f}")

    json.dump(rows, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    print(f"\n[batch] wrote {os.path.join(out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
