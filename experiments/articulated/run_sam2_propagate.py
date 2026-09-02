"""Propagate first-frame masks through an RBO sequence with SAM2, and score them.

Segmentation and propagation are SAM 2 (Ravi et al., Meta), via the
segment-anything-2-real-time fork vendored in this repo.

Initialised from masks on the first usable frame only, then propagated by SAM2
for the rest of the video.  Two modes:

  --init gt      one SAM2 object per GT part.  Measures how well SAM2 alone can
                 keep parts separated, which is the upper bound for any method
                 that gets its per-part masks this way.
  --init union   a single object covering the whole articulated body, the
                 realistic setting where nobody tells us the parts.

Writes an overlay video, per-frame masks, and per-part IoU against ground truth
so the propagation can be checked rather than assumed.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from point2pose.io.sources.dataset.sapien_reader import open_sequence
from experiments.articulated.run_rbo_part_discovery import find_init_frame, union_mask

COLORS = [(66, 135, 245), (214, 130, 60), (90, 190, 100),
          (170, 100, 220), (80, 200, 230), (120, 120, 250)]


def pick_points(rgb, segment_fn, pos=None, neg=None):
    """Click the object and see SAM2's mask update after every click.

    `segment_fn(points, labels) -> mask` re-prompts SAM2 with all accumulated
    clicks, so the overlay always shows what would actually be propagated.
    Nothing is committed until Enter.
    """
    pos, neg = list(pos or []), list(neg or [])
    base = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    state = {"mask": None, "busy": False}

    def recompute():
        if not pos:
            state["mask"] = None
            return
        pts = np.array(pos + neg, dtype=np.float32)
        lbl = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)
        state["busy"] = True
        try:
            state["mask"] = segment_fn(pts, lbl)
        except Exception as e:                       # keep the window alive
            print(f"[sam2] preview failed: {type(e).__name__}: {e}")
            state["mask"] = None
        state["busy"] = False

    def redraw():
        v = base.copy()
        m = state["mask"]
        if m is not None and m.sum() > 0:
            c = np.array([245, 135, 66], np.float32)
            sel = m > 0
            v[sel] = (0.45 * c + 0.55 * v[sel]).astype(np.uint8)
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(v, cnts, -1, (245, 135, 66), 2)
        for x, y in pos:
            cv2.circle(v, (int(x), int(y)), 6, (80, 230, 120), -1, cv2.LINE_AA)
            cv2.circle(v, (int(x), int(y)), 6, (20, 20, 20), 1, cv2.LINE_AA)
        for x, y in neg:
            cv2.circle(v, (int(x), int(y)), 6, (60, 60, 240), -1, cv2.LINE_AA)
            cv2.circle(v, (int(x), int(y)), 6, (20, 20, 20), 1, cv2.LINE_AA)
        bar = np.full((56, v.shape[1], 3), (25, 22, 18), np.uint8)
        cv2.putText(bar, "left = object    right = exclude    u = undo    "
                         "r = reset    ENTER = accept    q = cancel",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1,
                    cv2.LINE_AA)
        m = state["mask"]
        px = int(m.sum()) if m is not None else 0
        pct = 100.0 * px / (v.shape[0] * v.shape[1])
        msg = (f"{len(pos)}+ / {len(neg)}-   mask {px} px ({pct:.1f}%)"
               if px else f"{len(pos)}+ / {len(neg)}-   no mask yet - click the object")
        cv2.putText(bar, msg, (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (150, 220, 255) if px else (170, 170, 170), 1, cv2.LINE_AA)
        return np.vstack([v, bar])

    win = "click the object  (first frame)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, rgb.shape[1], rgb.shape[0] + 56)

    def on_mouse(ev, x, y, flags, _):
        if state["busy"]:
            return
        if ev == cv2.EVENT_LBUTTONDOWN:
            pos.append((float(x), float(y)))
        elif ev == cv2.EVENT_RBUTTONDOWN:
            neg.append((float(x), float(y)))
        else:
            return
        recompute()
        cv2.imshow(win, redraw())

    cv2.setMouseCallback(win, on_mouse)
    if pos:
        recompute()
    cv2.imshow(win, redraw())
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10):
            if not pos:
                print("[sam2] click at least one point on the object first")
                continue
            break
        if k == ord("u") and (pos or neg):
            (pos.pop() if pos else neg.pop())
            recompute(); cv2.imshow(win, redraw())
        if k == ord("r"):
            pos.clear(); neg.clear(); state["mask"] = None
            cv2.imshow(win, redraw())
        if k == ord("q"):
            pos, neg = [], []
            break
    cv2.destroyWindow(win)
    cv2.waitKey(1)
    return pos, neg


def build_predictor(checkpoint, model_cfg, device="cuda"):
    from sam2.build_sam import build_sam2_camera_predictor
    return build_sam2_camera_predictor(model_cfg, checkpoint, device=device)


def overlay(rgb, masks, iou_txt, frame_id, n_obj):
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for i, m in enumerate(masks):
        if m is None or m.sum() == 0:
            continue
        c = np.array(COLORS[i % len(COLORS)], np.float32)
        sel = m > 0
        vis[sel] = (0.45 * c + 0.55 * vis[sel]).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, tuple(int(x) for x in COLORS[i % len(COLORS)]), 2)
    bar = np.full((34, vis.shape[1], 3), (28, 24, 20), np.uint8)
    cv2.putText(bar, f"frame {frame_id:04d}   {n_obj} SAM2 objects", (10, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    if iou_txt:
        cv2.putText(bar, iou_txt, (250, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (150, 220, 255), 1, cv2.LINE_AA)
    return np.vstack([vis, bar])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-dir", required=True)
    ap.add_argument("--init", choices=["gt", "union", "click", "bbox"], default="gt",
                    help="gt = one object per GT part (upper bound); "
                         "union = whole body from GT masks (still GT); "
                         "click = whole body from YOUR clicks, no GT at all; "
                         "bbox = a single 2D box, the weakest useful prompt")
    ap.add_argument("--bbox", default=None,
                    help="'x0,y0,x1,y1'; if omitted, the box enclosing the GT "
                         "object mask on the init frame is used")
    ap.add_argument("--click", action="append", default=[],
                    help="positive prompt 'x,y'; repeatable")
    ap.add_argument("--click-neg", action="append", default=[],
                    help="negative prompt 'x,y'; repeatable")
    ap.add_argument("--pick", action="store_true",
                    help="open a window and click the object on the first frame")
    ap.add_argument("--out", default="debug/multi-parts/runs/sam2/propagate.mp4")
    ap.add_argument("--save-masks", default=None, help="npz of propagated masks")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--checkpoint", default="checkpoints/sam2.1/sam2.1_hiera_large.pt")
    ap.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--show", action="store_true",
                    help="watch the propagation live; q aborts, space pauses")
    ap.add_argument("--no-show", dest="show", action="store_false")
    ap.set_defaults(show=None)
    args = ap.parse_args()

    reader = open_sequence(args.seq_dir)
    parts = reader.get_object_names()
    frames = list(range(0, len(reader), args.stride))
    start = find_init_frame(reader, frames)
    frames = frames[start:]
    if args.max_frames:
        frames = frames[: args.max_frames]
    print(f"[sam2] {reader.get_video_name()}  {len(frames)} frames  "
          f"init on frame {frames[0]}  mode={args.init}  GT parts {parts}")

    predictor = build_predictor(args.checkpoint, args.model_cfg)
    show = args.show if args.show is not None else args.pick

    f0 = frames[0]
    rgb0 = reader.get_color(f0)
    predictor.load_first_frame(rgb0)

    kept = []
    if args.init == "bbox":
        if args.bbox:
            box = [float(v) for v in args.bbox.split(",")]
        else:
            # A box is far weaker supervision than a per-pixel mask -- it is what a
            # user would drag -- but note it is still derived from GT here, on the
            # init frame only.  Pass --bbox to remove GT from the input entirely.
            um = union_mask(reader, f0, reader.get_depth(f0))
            ys, xs = np.where(um > 0)
            box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        print(f"[sam2] bbox prompt {['%.0f' % v for v in box]}")
        _, _, logits = predictor.add_new_prompt(
            frame_idx=0, obj_id=0, bbox=np.array(box, dtype=np.float32))
        init_masks = [(logits[0, 0] > 0).cpu().numpy().astype(np.uint8)]
        kept = ["object"]
        print(f"[sam2] bbox mask covers {int(init_masks[0].sum())} px "
              f"({100*init_masks[0].mean():.1f}% of the image)")
    elif args.init == "click":
        pos = [tuple(float(v) for v in c.split(",")) for c in args.click]
        neg = [tuple(float(v) for v in c.split(",")) for c in args.click_neg]

        def segment_fn(pts, lbl):
            _, _, lg = predictor.add_new_prompt(
                frame_idx=0, obj_id=0, points=pts, labels=lbl,
                clear_old_points=True)
            return (lg[0, 0] > 0).cpu().numpy().astype(np.uint8)

        if args.pick or not pos:
            pos, neg = pick_points(rgb0, segment_fn, pos, neg)
        if not pos:
            raise SystemExit("no positive click given; use --click x,y or --pick")
        pts = np.array(pos + neg, dtype=np.float32)
        lbl = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)
        print(f"[sam2] prompting with {len(pos)} positive / {len(neg)} negative clicks")
        init_masks = [segment_fn(pts, lbl)]
        kept = ["object"]
        print(f"[sam2] accepted mask covers {int(init_masks[0].sum())} px "
              f"({100*init_masks[0].mean():.1f}% of the image)")
    else:
        if args.init == "gt":
            init_masks = [m for m in reader.get_masks(f0)]
            names = list(parts)
        else:
            init_masks = [union_mask(reader, f0, reader.get_depth(f0))]
            names = ["object"]
        for i, m in enumerate(init_masks):
            if m.sum() < 200:
                print(f"[sam2]   skipping {names[i]}: only {int(m.sum())} px")
                continue
            predictor.add_new_mask(frame_idx=0, obj_id=len(kept),
                                   mask=torch.from_numpy(m.astype(bool)))
            kept.append(names[i])
        init_masks = [m for m in init_masks if m.sum() >= 200]
    print(f"[sam2] initialised {len(kept)} objects: {kept}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = None
    ious, saved, hist = {n: [] for n in kept}, [], []

    for n, i in enumerate(frames):
        rgb = reader.get_color(i)
        if n == 0:
            masks = [m.astype(np.uint8) for m in init_masks]
        else:
            _, logits = predictor.track(rgb)
            masks = [(logits[k, 0] > 0).cpu().numpy().astype(np.uint8)
                     for k in range(logits.shape[0])]

        # score against GT for whichever GT part each object best matches
        gt = reader.get_masks(i)
        txt = []
        for k, name in enumerate(kept):
            if k >= len(masks):
                continue
            best, best_i = 0.0, None
            for gi, g in enumerate(gt):
                inter = np.logical_and(masks[k] > 0, g > 0).sum()
                union = np.logical_or(masks[k] > 0, g > 0).sum()
                iou = inter / union if union else 0.0
                if iou > best:
                    best, best_i = iou, parts[gi]
            ious[name].append(best)
            txt.append(f"{name}:{best:.2f}")
        iou_txt = "IoU " + " ".join(txt) if txt else ""

        canvas = overlay(rgb, masks, iou_txt, i, len(masks))
        if writer is None:
            h, w = canvas.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.fps, (w, h))
        writer.write(canvas)
        if show:
            cv2.imshow("SAM2 propagation  (q abort, space pause)", canvas)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                print("[sam2] aborted by user")
                break
            if k == ord(" "):
                while (cv2.waitKey(30) & 0xFF) != ord(" "):
                    pass
        if args.save_masks:
            saved.append(np.stack(masks) if masks else np.zeros((0,) + rgb.shape[:2], np.uint8))
        hist.append({"frame": int(i),
                     "iou": {nm: float(ious[nm][-1]) for nm in kept if ious[nm]}})
        if n % 25 == 0:
            print(f"  frame {i:4d}  {iou_txt}")

    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    print(f"\n[sam2] wrote {args.out}")
    for nm in kept:
        a = np.array(ious[nm])
        if a.size:
            print(f"  {nm:8s} IoU mean {a.mean():.3f}  median {np.median(a):.3f}  "
                  f"min {a.min():.3f}  final {a[-1]:.3f}")
    if args.save_masks:
        np.savez_compressed(args.save_masks, masks=np.stack(saved),
                            frames=np.array(frames[: len(saved)]),
                            names=np.array(kept))
        print(f"[sam2] wrote {args.save_masks}")
    json.dump({"sequence": reader.get_video_name(), "init": args.init,
               "objects": kept, "history": hist},
              open(os.path.splitext(args.out)[0] + "_report.json", "w"), indent=2)


if __name__ == "__main__":
    main()
