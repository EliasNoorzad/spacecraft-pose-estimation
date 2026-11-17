"""DLVS3-HST PNG/JSON -> YOLOv8 Pose with visibility and approximate boxes."""

import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
METADATA = ROOT / "0000000_metadata.json"
IMAGE_DIR = ROOT / "0000000"
OUTPUT = ROOT / "hst_yolo_pose_fixed"
NUM_KEYPOINTS = 37
BOX_PADDING = 0.10
SEED = 42


def frames_from(obj):
    if isinstance(obj, dict):
        if "ImagePath" in obj and "TargetKeypoints" in obj:
            yield obj
        else:
            for val in obj.values():
                yield from frames_from(val)
    elif isinstance(obj, list):
        for val in obj:
            yield from frames_from(val)


def index_pngs(root):
    by_stem = defaultdict(list)
    for p in root.rglob("*.png"):
        by_stem[p.stem].append(p)
    return by_stem


def locate_png(frame, by_stem):
    raw_path = str(frame["ImagePath"]).replace("\\", "/")
    expected = ROOT / Path(raw_path).with_suffix(".png")
    if expected.is_file():
        return expected
    candidates = by_stem.get(Path(raw_path).stem, [])
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous PNG match for {raw_path}: {candidates}")
    return None


def keypoint_data(frame, w, h):
    source = frame["TargetKeypoints"]
    if not isinstance(source, list) or len(source) != NUM_KEYPOINTS:
        raise ValueError("Expected exactly 37 TargetKeypoints")
    normalized = []
    positions = []
    n_invalid = 0
    for pt in source:
        try:
            x, y = float(pt["X"]), float(pt["Y"])
        except (KeyError, TypeError, ValueError):
            x = y = float("nan")
        if math.isfinite(x) and math.isfinite(y) and 0 <= x < w and 0 <= y < h:
            # v=1 means labeled/in frame; actual occlusion is NOT in the JSON.
            normalized.extend((x / w, y / h, 1))
            positions.append((x, y))
        else:
            # v=0 instructs YOLO to ignore this point in keypoint loss.
            normalized.extend((0.0, 0.0, 0))
            positions.append(None)
            n_invalid += 1
    return normalized, positions, n_invalid


def box_from_points(positions, w, h):
    valid = [pt for pt in positions if pt is not None]
    if len(valid) < 2:
        return None
    x0 = min(p[0] for p in valid)
    x1 = max(p[0] for p in valid)
    y0 = min(p[1] for p in valid)
    y1 = max(p[1] for p in valid)
    px = max(2.0, (x1 - x0) * BOX_PADDING)
    py = max(2.0, (y1 - y0) * BOX_PADDING)
    x0, y0 = max(0.0, x0 - px), max(0.0, y0 - py)
    x1, y1 = min(float(w), x1 + px), min(float(h), y1 + py)
    if x1 <= x0 or y1 <= y0:
        return None
    box = ((x0+x1)/(2*w), (y0+y1)/(2*h), (x1-x0)/w, (y1-y0)/h)
    return box


def main():
    if not METADATA.is_file() or not IMAGE_DIR.is_dir():
        raise SystemExit("Put this script beside 0000000_metadata.json and 0000000/ PNG folder")

    with METADATA.open("r", encoding="utf-8-sig") as f:
        frames = list(frames_from(json.load(f)))
    if not frames:
        raise SystemExit("No frames found in metadata")

    image_by_stem = index_pngs(IMAGE_DIR)
    items = []
    missing = bad = total_invalid = 0
    seen_paths = set()
    for frame in frames:
        image_path = locate_png(frame, image_by_stem)
        if image_path is None:
            missing += 1
            continue
        if image_path in seen_paths:
            raise ValueError(f"Two annotation frames refer to the same image: {image_path}")
        seen_paths.add(image_path)
        try:
            with Image.open(image_path) as im:
                w,h = im.size
            flat, positions, n_invalid = keypoint_data(frame, w, h)
            box_norm = box_from_points(positions, w, h)
            if box_norm is None:
                raise ValueError("Fewer than two usable projected keypoints")
            vals = [0, *box_norm, *flat]
            if len(vals) != 5 + NUM_KEYPOINTS*3:
                raise ValueError("Incorrect label length")
            label = str(vals[0]) + " " + " ".join(f"{v:.8f}" for v in vals[1:]) + "\n"
            # Round-trip verification ensures keypoint order and coordinates were not changed.
            output_values = [float(n) for n in label.split()]
            for i, xy in enumerate(positions):
                xp, yp, visibility = output_values[5+3*i:5+3*i+3]
                if xy is not None:
                    if visibility != 1 or abs(xp*w-xy[0]) > 0.0001 or abs(yp*h-xy[1]) > 0.0001:
                        raise ValueError(f"Round-trip mismatch at keypoint {i}")
                elif (xp,yp,visibility) != (0,0,0):
                    raise ValueError(f"Invalid point {i} was not masked")
            total_invalid += n_invalid
            items.append((image_path, label))
        except (OSError, ValueError, TypeError) as exc:
            bad += 1
            print(f"SKIP {image_path}: {exc}")

    if missing or bad:
        raise SystemExit(f"Stopped before writing: missing matches={missing}, invalid annotations={bad}. Fix these first.")
    if not items:
        raise SystemExit("No usable image-label pairs")

    random.Random(SEED).shuffle(items)
    n = len(items)
    a = int(0.8*n)
    b = a+int(0.1*n)
    splits = {"train":items[:a], "val":items[a:b], "test":items[b:]}

    for split, subset in splits.items():
        idest = OUTPUT/"images"/split
        ldest = OUTPUT/"labels"/split
        idest.mkdir(parents=True,exist_ok=True)
        ldest.mkdir(parents=True,exist_ok=True)
        for image_path, label in subset:
            dest = idest / image_path.name
            if dest.exists():
                # A previous run may have reused the output: overwrite to avoid stale content.
                dest.unlink()
            shutil.copy2(image_path,dest)
            (ldest / (image_path.stem+".txt")).write_text(label,encoding="utf-8")

    (OUTPUT/"data.yaml").write_text(
        f'path: "{OUTPUT.resolve().as_posix()}"\n'
        'train: images/train\nval: images/val\ntest: images/test\n'
        'names:\n  0: HST\nkpt_shape: [37, 3]\n',
        encoding="utf-8",
    )
    print(f"Done: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    print(f"Keypoints masked as invalid/out-of-frame: {total_invalid}")
    print("All generated labels passed pixel-coordinate round-trip verification")
    print(f"Dataset: {OUTPUT}")
    print("Important: in-frame does NOT guarantee unoccluded; boxes are approximations")


if __name__ == "__main__":
    main()
