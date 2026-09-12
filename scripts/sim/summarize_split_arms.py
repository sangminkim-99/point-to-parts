#!/usr/bin/env python
"""One table over the matched split-gate arms in results/split_purity_v1.

Per (case, arm): parts vs GT, purity, joint-coverage line, refusal counters,
and the birth purity of every accepted split's children (from the split-diag
dump + GT track labels). Reads stdouts and *_splitdiag.json; runs nothing.
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/split_purity_v1")


def birth_purity(diag):
    d = json.load(open(diag))
    lab, names = d["gt_lab"], d["gt_parts"]
    out = []
    for sp in d["splits"]:
        if "children" not in sp:
            continue
        for c in sp["children"]:
            l = np.array([lab[i] if i < len(lab) else -1 for i in c["idx"]])
            l = l[l >= 0]
            pur = float((l == np.bincount(l).argmax()).mean()) if l.size else float("nan")
            out.append((sp["frame"], c["n"], pur))
    return out


rows = {}
for so in sorted(glob.glob(str(ROOT / "*.stdout"))):
    name = Path(so).stem
    m = re.match(r"(.+)_(base|perchild|refine|both)$", name)
    if not m:
        continue
    case, arm = m.groups()
    txt = open(so).read()
    ev = re.search(r"\[eval\] \S+: (\d+) parts vs (\d+) GT, covered (\d+/\d+), purity ([\d.]+)%", txt)
    cov = re.search(r"joint coverage: (.*)", txt)
    refine = re.search(r"coassoc refinement: (.*)", txt)
    sep = re.search(r"\[replay\] (\d+) splits refused: the two groups", txt)
    splits = re.search(r"splits (\[[^\]]*\])", txt)
    diag = ROOT / f"{name}_splitdiag.json"
    bp = birth_purity(diag) if diag.exists() else []
    rows[(case, arm)] = {
        "eval": ev.groups() if ev else None, "coverage": cov.group(1) if cov else "",
        "splits": splits.group(1) if splits else "", "refine": refine.group(1) if refine else "",
        "sep_refused": sep.group(1) if sep else "0",
        "births": " ".join(f"f{f}:{n}pts@{p:.2f}" for f, n, p in bp)}

cases = sorted({c for c, _ in rows})
for case in cases:
    print(f"\n=== {case}")
    for arm in ("base", "perchild", "refine", "both"):
        r = rows.get((case, arm))
        if r is None:
            continue
        e = r["eval"]
        ev = f"{e[0]} parts vs {e[1]} GT, covered {e[2]}, purity {e[3]}%" if e else "no eval"
        print(f"  {arm:>8}: {ev}; splits {r['splits']}; sep-refused {r['sep_refused']}"
              + (f"; refine: {r['refine']}" if r["refine"] else ""))
        print(f"            joints: {r['coverage'] or '-'}")
        print(f"            births: {r['births'] or '-'}")
