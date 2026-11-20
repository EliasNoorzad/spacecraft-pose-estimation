"""Convert DLVS3-HST PNG images and annotations to YOLOv8 Pose format.

Use --root for numbered groups or --images and --metadata for a single file.
When image stems collide, filename_map.json maps each renamed output stem to
its source metadata filename, original ImagePath, and zero-based frame index.
"""

import argparse
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

NUM_KEYPOINTS = 37
BOX_PADDING = 0.10
SEED = 42


def frames_from(obj):
    if isinstance(obj, dict):
        if "ImagePath" in obj or "TargetKeypoints" in obj:
            yield obj
        else:
            for val in obj.values():
                yield from frames_from(val)
    elif isinstance(obj, list):
        for val in obj:
            yield from frames_from(val)


def index_pngs(folder):
    by_stem = defaultdict(list)
    for path in folder.rglob("*"):
        if path.suffix.lower() == ".png" and path.is_file():
            if not path.resolve().is_relative_to(folder):
                raise ValueError(f"PNG outside its image folder: {path}")
            by_stem[path.stem].append(path)
    return by_stem


def locate_png(raw_path, by_stem, folder, metadata_dir, grouped):
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("Invalid ImagePath")
    ref = Path(raw_path.replace("\\", "/")).with_suffix(".png")
    candidates = by_stem.get(ref.stem, [])
    if not grouped:
        expected = metadata_dir / ref
        if expected.is_file():
            return expected

    # Resolve paths against this group's index only, even for absolute references.
    parts = ref.parts
    if grouped:
        if ".." in parts:
            raise ValueError(f"Parent traversal in ImagePath: {raw_path}")
        for i, part in enumerate(parts[:-1]):
            if len(part) == 7 and part.isdecimal():
                if part != folder.name:
                    raise ValueError(f"ImagePath refers to group {part}, expected {folder.name}: {raw_path}")
                parts = parts[i + 1:]
                break
    expected = folder.joinpath(*parts)
    if expected in candidates:
        return expected
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous PNG match for {raw_path}: {candidates}")
    return candidates[0] if candidates else None


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


def make_label(frame, image_path):
    with Image.open(image_path) as im:
        if im.format != "PNG":
            raise ValueError("Expected a PNG image")
        w, h = im.size
        im.verify()
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
    return label, n_invalid


@dataclass
class Item:
    image: Path
    label: str
    metadata: Path
    frame_index: int
    reference: str


def collect(groups, grouped, problems):
    items = []
    seen = set()
    frames = matched = missing = masked = 0
    for metadata, folder in groups:
        try:
            if grouped and (not folder.name.isdecimal() or metadata.resolve() != metadata):
                raise ValueError(f"Expected a numbered metadata file in the dataset root: {metadata}")
            with metadata.open(encoding="utf-8-sig") as f:
                annotations = list(frames_from(json.load(f)))
            frames += len(annotations)
            if not annotations:
                raise ValueError("No annotated frames in metadata")
            if not folder.is_dir() or (grouped and folder.resolve() != folder):
                raise ValueError(f"Missing or linked image folder: {folder}")
            index = index_pngs(folder)
        except (OSError, ValueError) as exc:
            problems.append(f"{metadata}: {exc}")
            continue

        group_matches = set()
        for i, frame in enumerate(annotations):
            ref = frame.get("ImagePath")
            try:
                image = locate_png(ref, index, folder, metadata.parent, grouped)
                if image is None:
                    missing += 1
                    raise ValueError("Missing PNG")
                path = image.resolve()
                if path in seen:
                    raise ValueError(f"Duplicate reference to {image}")
                seen.add(path)
                group_matches.add(image)
                matched += 1
                label, n_invalid = make_label(frame, image)
                masked += n_invalid
                items.append(Item(image, label, metadata, i, ref))
            except (OSError, ValueError, TypeError, KeyError, SyntaxError, Image.DecompressionBombError) as exc:
                problems.append(f"{metadata.name}, frame {i}, ImagePath={ref!r}: {exc}")
        if grouped:
            for paths in index.values():
                for path in paths:
                    if path not in group_matches:
                        problems.append(f"PNG without a matching annotation: {path}")

    print(f"Annotated frames: {frames}; uniquely matched PNG images: {matched}; missing images: {missing}")
    print(f"Missing/invalid input issues: {len(problems)}")
    if problems:
        for problem in problems:
            print(problem)
        raise SystemExit("Coordinate verification incomplete; generated 0 labels. Fix the input before retrying.")
    print(f"Keypoints masked as invalid/out-of-frame: {masked}")
    return items


def write_dataset(items, dest, output):
    # Reserve original stems across all splits, including on case-insensitive systems.
    counts = Counter(item.image.stem.casefold() for item in items)
    used = set(counts)
    renamed = {}
    random.Random(SEED).shuffle(items)
    n = len(items)
    a = int(0.8 * n)
    b = a + int(0.1 * n)
    splits = {"train": items[:a], "val": items[a:b], "test": items[b:]}
    for split, subset in splits.items():
        images = dest / "images" / split
        labels = dest / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for item in subset:
            stem = item.image.stem
            if counts[stem.casefold()] > 1:
                group = item.metadata.stem.removesuffix("_metadata")
                prefix = f"{group}__{stem}"
                stem = prefix
                suffix = 2
                while stem.casefold() in used:
                    stem = f"{prefix}__{suffix}"
                    suffix += 1
                used.add(stem.casefold())
                renamed[stem] = {"metadata": item.metadata.name, "frame_index": item.frame_index,
                                 "ImagePath": item.reference, "split": split}
            shutil.copy2(item.image, images / (stem + item.image.suffix))
            (labels / (stem + ".txt")).write_text(item.label, encoding="utf-8")

    if renamed:
        (dest / "filename_map.json").write_text(json.dumps(renamed, indent=2), encoding="utf-8")
    (dest / "data.yaml").write_text(
        f'path: {json.dumps(output.as_posix())}\n'
        'train: images/train\nval: images/val\ntest: images/test\n'
        'names:\n  0: HST\nkpt_shape: [37, 3]\n',
        encoding="utf-8",
    )
    return {split: len(subset) for split, subset in splits.items()}, len(renamed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--root", type=Path, help="DLVS3-HST root with numbered image folders and *_metadata.json")
    mode.add_argument("--images", type=Path, help="PNG folder for single-metadata mode")
    parser.add_argument("--metadata", type=Path, help="Single annotation file (requires --images)")
    parser.add_argument("--output", type=Path, required=True, help="New or empty YOLO dataset directory")
    args = parser.parse_args(argv)
    if bool(args.images) != bool(args.metadata):
        parser.error("Use --root alone or specify both --images and --metadata")

    output = args.output.expanduser().resolve()
    problems = []
    grouped = args.root is not None
    if grouped:
        root = args.root.expanduser().resolve()
        if not root.is_dir():
            parser.error(f"Dataset root not found: {root}")
        files = sorted(root.glob("*_metadata.json"))
        groups = [(p, root / p.name.removesuffix("_metadata.json")) for p in files]
        folders = {folder for _, folder in groups}
        for folder in root.iterdir():
            if folder.is_dir() and folder.name.isdecimal() and folder not in folders:
                problems.append(f"No metadata for image folder: {folder}")
        if not files:
            problems.append(f"No *_metadata.json files found in {root}")
    else:
        groups = [(args.metadata.expanduser().resolve(), args.images.expanduser().resolve())]

    print(f"Metadata files discovered: {sum(metadata.is_file() for metadata, _ in groups)}")
    if any(output.is_relative_to(folder) for _, folder in groups):
        parser.error("Output must be outside the source image folders")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Output must be new or empty to protect existing datasets: {output}")
    items = collect(groups, grouped, problems)
    if grouped and (len(groups) != 10 or len(items) != 10000):
        print(f"Warning: this input contains {len(groups)} groups and {len(items)} pairs; "
              "the complete dataset is expected to contain 10 groups and 10000 images.")

    # Publish only a complete conversion. A failed copy leaves the destination untouched.
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".hst-convert-") as tmp:
        dest = Path(tmp) / "dataset"
        totals, renamed = write_dataset(items, dest, output)
        if output.exists():
            output.rmdir()  # Succeeds only for an empty directory.
        dest.rename(output)
    print(f"Generated YOLO labels: {sum(totals.values())}")
    print(f"Images: train={totals['train']} val={totals['val']} test={totals['test']}")
    print(f"Filename collisions: {renamed} images renamed" + (" (see filename_map.json)" if renamed else ""))
    print("All generated labels passed pixel-coordinate round-trip verification")
    print(f"Dataset: {output}")


if __name__ == "__main__":
    main()
