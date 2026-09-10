"""Build and validate a small hand-free RGB-D development suite (SAPIEN 3)."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def validate_sequence(path, expected_parts):
    """Check stored files, GT transforms, coverage and return motion diagnostics."""
    meta = json.loads((path / "meta.json").read_text())
    n, k = meta["frames"], len(meta["parts"])
    assert k == expected_parts, (path, k, expected_parts)
    poses = np.load(path / "poses.npz")
    T = poses["T_cam_part"]
    assert T.shape == (n, k, 4, 4) and np.isfinite(T).all()
    assert np.allclose(T[..., 3, :], [0, 0, 0, 1])
    R = T[..., :3, :3]
    assert np.allclose(R.swapaxes(-1, -2) @ R, np.eye(3), atol=2e-5)
    assert np.allclose(np.linalg.det(R), 1, atol=2e-5)
    assert poses["joint_states"].shape == (n, len(meta["joints"]))
    counts, border, tiles = [], [], []
    for folder in ("rgb", "depth", "seg"):
        assert len(list((path / folder).glob("*.png"))) == n
    probes = set(np.linspace(0, n - 1, 6).astype(int))
    for t in range(n):
        name = f"{t:06d}.png"
        rgb = cv2.imread(str(path / "rgb" / name))
        depth = cv2.imread(str(path / "depth" / name), -1)
        seg = cv2.imread(str(path / "seg" / name), -1)
        assert rgb is not None and depth is not None and seg is not None
        assert list(depth.shape) == meta["image_size"] == list(seg.shape)
        assert depth.dtype == np.uint16 and seg.dtype == np.uint8
        assert set(np.unique(seg)) <= set(range(k)) | {255}
        mask = seg != 255
        assert mask.any() and np.all(depth[mask] > 0)
        counts.append([int((seg == p).sum()) for p in range(k)])
        border.append(int(mask[0].sum() + mask[-1].sum() + mask[:, 0].sum() + mask[:, -1].sum()))
        if t in probes:
            tile = cv2.resize(rgb, (320, 240))
            cv2.putText(tile, f"{path.name}  f{t}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 180, 255), 1)
            tiles.append(tile)
    counts = np.array(counts)
    assert np.all(counts.max(axis=0) > 50), "invisible GT motion group"
    assert max(border) == 0, "object clipped by image boundary"
    assert meta["pixel_reprojection_max_px"] < .1
    cv2.imwrite(str(path / "preview.jpg"), np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
    world = poses["cam_poses"][:, None] @ T
    return {"frames": n, "parts": k, "min_visible_pixels": counts.min(axis=0).tolist(),
            "visible_frames": (counts > 50).sum(axis=0).tolist(),
            "border_pixels_max": max(border),
            "pixel_reprojection_max_px": meta["pixel_reprojection_max_px"],
            "world_part_translation_range_m": np.ptp(world[..., :3, 3], axis=0).tolist(),
            "joint_travel": np.ptp(poses["joint_states"], axis=0).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sim/partnet_dev_v1.json")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"config": config, "cases": []}
    for case in config["cases"]:
        dest = args.out / case["name"]
        options = {**config["defaults"], **case["args"]}
        cmd = [sys.executable, str(ROOT / "scripts/sim/render_partnet_sequence.py"),
               "--model-dir", str(args.assets / case["model"]), "--out", str(dest)]
        for key, value in options.items():
            if value is False:
                continue
            cmd.append("--" + key)
            if value is not True:
                cmd.append(str(value))
        if not args.validate_only and not (dest / "meta.json").exists():
            subprocess.run(cmd, check=True)
        elif not args.validate_only:
            # Resume only identical settings; do not silently reuse another dataset.
            previous = dest / "render_command.json"
            if not previous.exists() or json.loads(previous.read_text()) != cmd:
                raise RuntimeError(f"{dest}: existing sequence has different/unknown settings")
        metrics = validate_sequence(dest, case["expected_parts"])
        (dest / "render_command.json").write_text(json.dumps(cmd, indent=2) + "\n")
        manifest["cases"].append({"name": case["name"], "command": cmd, **metrics})
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"[dev-set] validated {case['name']}: {metrics}", flush=True)


if __name__ == "__main__":
    main()
