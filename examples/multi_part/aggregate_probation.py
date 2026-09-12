#!/usr/bin/env python
"""Assemble one comparison table from every *_lifecycle.json in a results dir.

The runner overwrites its own comparison.json per invocation, so when the matrix
is filled across several invocations this rebuilds the whole picture from the
per-cell ledgers. Read-only: it never runs the tracker.
"""
import argparse
import json
from pathlib import Path

DEFAULT_DIR = "/home/smkim/workspace/code/point-to-pose-model-based/results/newborn_probation_v2"


def row(life):
    parts = life.get("parts", [])
    covs = [p["coverage"] for p in parts if p.get("alive_frames")]
    lifetimes = [p["lifetime"] for p in parts]
    return {
        "frames": life.get("frames"),
        "final_parts": life.get("final_parts"),
        "ids_born": life.get("ids_born"),
        "ids_died": life.get("ids_died"),
        "ids_surviving": life.get("ids_surviving"),
        "splits": len(life.get("split_log", [])),
        "min_lifetime": min(lifetimes) if lifetimes else None,
        "coverage_min": round(min(covs), 4) if covs else None,
        "coverage_mean": round(sum(covs) / len(covs), 4) if covs else None,
        "unearned_verdicts": (life.get("verdicts_on_unearned_evidence") or {}).get("total"),
        "withheld_checks": (life.get("probation_withheld") or {}).get("checks"),
        "withheld_parts": (life.get("probation_withheld") or {}).get("part_ids"),
        "merge_probation": (life.get("config") or {}).get("merge_probation"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    args = ap.parse_args()
    d = Path(args.dir)
    cells = {}
    for f in sorted(d.glob("*_lifecycle.json")):
        name = f.name[:-len("_lifecycle.json")]
        seq, _, variant = name.rpartition("__")
        try:
            cells.setdefault(seq, {})[variant] = row(json.loads(f.read_text()))
        except Exception as e:  # a truncated/mid-write file
            cells.setdefault(seq, {})[variant] = {"error": str(e)}
    (d / "comparison_full.json").write_text(json.dumps(cells, indent=2, allow_nan=False))

    hdr = ("seq/variant", "born", "died", "surv", "splits", "min_life",
           "cov_min", "cov_mean", "unearned", "withheld")
    print("%-26s %4s %4s %-10s %6s %8s %7s %8s %8s %8s" % hdr)
    print("-" * 100)
    for seq in sorted(cells):
        for variant in ("baseline", "probation"):
            r = cells[seq].get(variant)
            if not r:
                continue
            if "error" in r:
                print("%-26s  (in progress / %s)" % (f"{seq}/{variant}", r["error"][:30]))
                continue
            print("%-26s %4s %4s %-10s %6s %8s %7s %8s %8s %8s" % (
                f"{seq}/{variant}", r["ids_born"], r["ids_died"],
                str(r["ids_surviving"])[:10], r["splits"], r["min_lifetime"],
                r["coverage_min"], r["coverage_mean"],
                r["unearned_verdicts"], r["withheld_checks"]))
    print(f"\n[wrote] {d / 'comparison_full.json'}")


if __name__ == "__main__":
    main()
