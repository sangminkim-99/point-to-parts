"""Unpack an RBO archive into a form RBOReader can read efficiently.

RBO is read natively; the only thing changed on disk is depth.  RBO stores depth
as plain-text float matrices -- 6.9 MB and ~47 ms per frame, against 0.1 MB and
~2 ms for a uint16 PNG, so keeping 154 sequences in the original form would cost
~236 GB and a 22x decode penalty on every run.  This extracts an archive, writes
`depth_png/` beside the original data, and drops the text depth.

Everything else (camera_rgb/, the CSVs, the YAML) is left exactly as shipped.
"""

import argparse
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd


def cache_depth(seq_dir, keep_text=False, quiet=False):
    src = os.path.join(seq_dir, "camera_depth_registered")
    dst = os.path.join(seq_dir, "depth_png")
    if not os.path.isdir(src):
        if os.path.isdir(dst):
            return 0  # already prepared and text already dropped
        raise FileNotFoundError(f"no depth directory under {seq_dir}")

    os.makedirs(dst, exist_ok=True)
    files = sorted(f for f in os.listdir(src) if f.endswith(".txt"))
    n = 0
    for f in files:
        out = os.path.join(dst, os.path.splitext(f)[0] + ".png")
        if os.path.exists(out):
            continue
        d = pd.read_csv(
            os.path.join(src, f), sep=r"\s+", header=None, dtype=np.float32
        ).to_numpy()
        mm = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0
        cv2.imwrite(out, np.clip(mm, 0, 65535).astype(np.uint16))
        n += 1
        if not quiet and n % 50 == 0:
            print(f"    {n}/{len(files)}", flush=True)
    if not keep_text:
        shutil.rmtree(src)
    return n


def prepare(archive, out_root, keep_text=False, quiet=False):
    name = os.path.basename(archive)[: -len(".tar.gz")]
    seq_dir = os.path.join(out_root, name)
    if not os.path.isdir(seq_dir):
        os.makedirs(out_root, exist_ok=True)
        subprocess.run(["tar", "xzf", archive, "-C", out_root], check=True)
    n = cache_depth(seq_dir, keep_text, quiet)
    return seq_dir, n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archives", nargs="+", help="raw/<seq>_o.tar.gz")
    ap.add_argument("--out-root", required=True, help="where sequences are unpacked")
    ap.add_argument("--keep-text", action="store_true",
                    help="keep the original text depth (costs ~1.5 GB per sequence)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    for a in args.archives:
        name = os.path.basename(a)
        try:
            seq_dir, n = prepare(a, args.out_root, args.keep_text, args.quiet)
            print(f"{name}: {n} depth frames cached -> {seq_dir}", flush=True)
        except Exception as e:
            print(f"{name}: FAILED ({type(e).__name__}: {e})", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
