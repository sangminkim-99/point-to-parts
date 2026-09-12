#!/usr/bin/env python
"""Matched baseline-vs-probation replay for the newborn-merge experiment.

Runs `examples.multi_part.replay` once per (sequence, variant) cell, where the
only thing that changes between the two variants is `merge_probation`:

    baseline   --set merge_probation=false   (the shipped default)
    probation  --set merge_probation=true

Reprojection gating stays ON for every cell (`--config reprojection_split.yaml`),
per the prior ablation. The comparison is read from the machine-readable
lifecycle ledger `--dump-lifecycle` writes -- identity survival, true observed
coverage, fragmentation and the probation bookkeeping -- NOT from the final part
count, which cannot tell a part that was discovered and kept from one that was
discovered, lost and rediscovered.

Sequences:
  * RealSense clips (take01, lift01): no ground truth, so only survival/coverage
    are reportable.
  * Sim controls carry GT (poses.npz + meta.json). laptop_orbit is a RIGID
    control -- the whole object orbits with no real joint, so any split SHOULD
    merge back. laptop_hinge is an ATTACHED revolute and storage_slide an
    attached prismatic -- a real joint that should survive. These are the
    rigid/attached controls the experiment brief asks for.

This runner never edits tracker code and never enables probation globally: it
only passes --set per cell. Default off remains the shipped behaviour.

Usage (from the repo root, with the point2pose_model env on PATH):

    P=$HOME/miniconda3/envs/point2pose_model
    env CUDA_HOME=$P PATH=$P/bin:$PATH $P/bin/python \
        -m examples.multi_part.run_probation_experiment \
        --sequences take01 lift01 laptop_orbit laptop_hinge

Add --dry-run to print the commands without running anything (no GPU).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Subprocesses run from the runner's OWN repo root, so the tracker code under
# test is this checkout's (a worktree, when run from one), not another copy that
# happens to sit on PATH. Inputs and artifacts are absolute under the main
# checkout's ignored results/ tree, which is where the reviewer looks and where
# the large gitignored sim inputs and asset dirs already live.
RUN_ROOT = Path(__file__).resolve().parents[2]
MAIN = Path("/home/smkim/workspace/code/point-to-pose-model-based")
DEFAULT_OUT = MAIN / "results" / "newborn_probation_v2"
DEFAULT_SIM_ROOT = MAIN / "results" / "partnet_dev_v1"
REALSENSE_ROOT = Path("/home/smkim/data/mp")

# name -> (kind, resolver). RealSense clips have no GT; sim controls do.
KNOWN = {
    "take01":       ("realsense", lambda a: REALSENSE_ROOT / "take01"),
    "lift01":       ("realsense", lambda a: REALSENSE_ROOT / "lift01"),
    "laptop_orbit": ("sim_rigid", lambda a: Path(a.sim_root) / "laptop_orbit"),
    "laptop_hinge": ("sim_revolute", lambda a: Path(a.sim_root) / "laptop_hinge"),
    "storage_slide": ("sim_prismatic", lambda a: Path(a.sim_root) / "storage_slide"),
    "laptop_static": ("sim_rigid", lambda a: Path(a.sim_root) / "laptop_static"),
}

VARIANTS = {
    "baseline": ["--set", "merge_probation=false"],
    "probation": ["--set", "merge_probation=true"],
}


def replay_cmd(seq_dir, out_mp4, lifecycle_json, variant, args):
    cmd = [sys.executable, "-u", "-m", "examples.multi_part.replay",
           "--seq-dir", str(seq_dir),
           "--method", "naive",
           "--config", "reprojection_split.yaml",
           "--stride", str(args.stride),
           "--view", "0", "--hyp-panel", "0",
           "--no-eval", "--vis", "clean",
           "--out", str(out_mp4),
           "--dump-lifecycle", str(lifecycle_json)]
    if args.probation_min_obs is not None:
        cmd += ["--set", f"probation_min_obs={args.probation_min_obs}"]
    cmd += VARIANTS[variant]
    if args.max_frames:
        cmd += ["--max-frames", str(args.max_frames)]
    return cmd


def summarise(life):
    """The report line for one lifecycle ledger: survival, coverage, probation."""
    parts = life.get("parts", [])
    covs = [p["coverage"] for p in parts if p.get("alive_frames")]
    survivors = [p for p in parts if p["survived"]]
    died = [p for p in parts if not p["survived"]]
    lifetimes = {p["part_id"]: p["lifetime"] for p in parts}
    return {
        "frames": life.get("frames"),
        "final_parts": life.get("final_parts"),
        "ids_born": life.get("ids_born"),
        "ids_died": life.get("ids_died"),
        "ids_surviving": life.get("ids_surviving"),
        # fragmentation: how many splits fired across the clip
        "splits": len(life.get("split_log", [])),
        "unearned_verdicts": life.get("verdicts_on_unearned_evidence"),
        "probation_withheld": life.get("probation_withheld"),
        # true observed coverage (observed frames / alive frames), min/mean:
        # 1.00 everywhere means coverage does not discriminate on this clip
        "coverage_min": round(min(covs), 4) if covs else None,
        "coverage_mean": round(sum(covs) / len(covs), 4) if covs else None,
        "survivor_ids": [p["part_id"] for p in survivors],
        "died_ids": [p["part_id"] for p in died],
        "min_lifetime": min(lifetimes.values()) if lifetimes else None,
        "config": life.get("config"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequences", nargs="+", default=["take01", "lift01"],
                    help="names from KNOWN, or absolute seq-dirs")
    ap.add_argument("--variants", nargs="+", default=["baseline", "probation"],
                    choices=list(VARIANTS))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--sim-root", default=str(DEFAULT_SIM_ROOT))
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--probation-min-obs", type=int, default=None,
                    help="override probation_min_obs for both variants' config")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands and exit; runs nothing")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    matrix = {}
    for name in args.sequences:
        if name in KNOWN:
            kind, resolve = KNOWN[name]
            seq_dir = resolve(args)
        else:
            kind, seq_dir = "custom", Path(name).expanduser()
            name = seq_dir.name
        if not seq_dir.exists():
            print(f"[skip] {name}: {seq_dir} does not exist")
            continue
        for variant in args.variants:
            tag = f"{name}__{variant}"
            mp4 = out_root / f"{tag}.mp4"
            life = out_root / f"{tag}_lifecycle.json"
            cmd = replay_cmd(seq_dir, mp4, life, variant, args)
            print(f"\n=== {tag} ({kind}) ===")
            print(" ".join(cmd))
            if args.dry_run:
                continue
            t0 = time.time()
            proc = subprocess.run(cmd, cwd=str(RUN_ROOT))
            dt = time.time() - t0
            entry = {"kind": kind, "seq_dir": str(seq_dir),
                     "returncode": proc.returncode, "wall_s": round(dt, 1),
                     "lifecycle_json": str(life)}
            if proc.returncode == 0 and life.exists():
                entry["summary"] = summarise(json.loads(life.read_text()))
            else:
                entry["error"] = "replay failed or no lifecycle written"
            matrix.setdefault(name, {})[variant] = entry
            print(f"[done] {tag} rc={proc.returncode} {dt:.1f}s")

    if args.dry_run:
        return
    report = out_root / "comparison.json"
    report.write_text(json.dumps(matrix, indent=2, allow_nan=False))
    print(f"\n[report] {report}")
    _print_table(matrix)


def _print_table(matrix):
    print("\n" + "=" * 78)
    print("BASELINE vs PROBATION -- identity survival, coverage, probation")
    print("=" * 78)
    for name, variants in matrix.items():
        print(f"\n{name}")
        for variant, entry in variants.items():
            s = entry.get("summary")
            if not s:
                print(f"  {variant:10s} FAILED ({entry.get('error')})")
                continue
            uv = s["unearned_verdicts"] or {}
            pw = s["probation_withheld"] or {}
            print(f"  {variant:10s} "
                  f"born={s['ids_born']} died={s['ids_died']} "
                  f"survive={s['ids_surviving']} splits={s['splits']} "
                  f"min_life={s['min_lifetime']} "
                  f"cov[min/mean]={s['coverage_min']}/{s['coverage_mean']} "
                  f"unearned={uv.get('total')} "
                  f"withheld={pw.get('checks')}")


if __name__ == "__main__":
    main()
