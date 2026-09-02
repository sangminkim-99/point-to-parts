"""Click the object once on the first frame; SAM 2 propagates it through a recording.

    python -m examples.multi_part.annotate --seq-dir <recording>

Left click adds a positive point, right click a negative one, 's' propagates.
Writes <recording>/masks.npz, which replay.py picks up automatically.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from examples.multi_part.recording import Recording


def flatten(logits, shape):
    """SAM2 returns [N,1,H,W] logits; we want one binary mask of the whole object."""
    if logits is None:
        return None
    m = np.asarray(logits.detach().cpu() if hasattr(logits, "detach") else logits)
    m = m[:, 0] if m.ndim == 4 else (m[None] if m.ndim == 2 else m)
    m = (m > 0.0).any(axis=0).astype(np.uint8)
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-dir", required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--sam-checkpoint",
                    default="checkpoints/sam2.1/sam2.1_hiera_large.pt")
    ap.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    args = ap.parse_args()

    from point2pose.modules.segmenter.sam2_real_time_segmenter import (
        Sam2RealTimeSegmenter)

    r = Recording(args.seq_dir)
    rgb0 = r.get_color(0)
    disp0 = cv2.cvtColor(rgb0, cv2.COLOR_RGB2BGR)
    pts, lbls = [], []
    sam = Sam2RealTimeSegmenter({"model_cfg": args.sam_config,
                                 "checkpoint": args.sam_checkpoint,
                                 "device": "cuda"})
    preview = [None]

    def on_mouse(event, x, y, _f, _p):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append([x, y]); lbls.append(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            pts.append([x, y]); lbls.append(0)
        else:
            return
        try:
            preview[0] = flatten(sam.preview(rgb0, [pts], [lbls]), rgb0.shape[:2])
        except Exception as exc:
            print(f"[sam] preview failed: {exc}")

    cv2.namedWindow("annotate")
    cv2.setMouseCallback("annotate", on_mouse)
    print("left click +, right click -, 's' to propagate, 'q' to quit")
    while True:
        view = disp0.copy()
        if preview[0] is not None:
            ov = np.zeros_like(view); ov[preview[0] > 0] = (60, 200, 60)
            view = cv2.addWeighted(view, 1.0, ov, 0.45, 0)
        for (x, y), l in zip(pts, lbls):
            if l == 1:
                cv2.circle(view, (x, y), 5, (60, 220, 60), -1)
            else:
                cv2.drawMarker(view, (x, y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
        cv2.putText(view, "L +  R -   's' propagate   'q' quit", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("annotate", view)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("q"):
            cv2.destroyAllWindows()
            return
        if k == ord("s") and any(l == 1 for l in lbls):
            break
    cv2.destroyAllWindows()

    sam.clear_input_objects()
    sam.add_input_object(pts, lbls)
    sam.initialize(rgb0)
    frames, masks = [], []
    idx = list(range(0, len(r), args.stride))
    for n, i in enumerate(idx):
        rgb = r.get_color(i)
        _, logits = sam.segment(rgb)
        m = flatten(logits, rgb.shape[:2])
        if m is None:
            m = np.zeros(rgb.shape[:2], np.uint8)
        frames.append(i); masks.append(m)
        if n % 20 == 0:
            print(f"  {n}/{len(idx)}  mask {int(m.sum())} px")
        vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ov = np.zeros_like(vis); ov[m > 0] = (60, 200, 60)
        cv2.imshow("propagate", cv2.addWeighted(vis, 1.0, ov, 0.45, 0))
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break
    cv2.destroyAllWindows()

    out = Path(args.seq_dir).expanduser() / "masks.npz"
    np.savez_compressed(out, frames=np.array(frames),
                        masks=np.stack(masks).astype(np.uint8))
    empty = sum(1 for m in masks if m.sum() < 200)
    print(f"wrote {out}  ({len(masks)} frames, {empty} nearly empty)")


if __name__ == "__main__":
    main()
