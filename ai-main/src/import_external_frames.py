"""Import external dataset frames into this project's structure.

Expected external layout (most common):
- <SRC_ROOT>/<class>/<video_id>/*.(jpg|jpeg|png)

What this script does:
- For each <class>/<video_id>, pick 5 evenly-spaced frames (sorted by name)
- Resize to 150x150 (per project config)
- Save into `data/processed/frames/<class>/<video_id>_f{i}.jpg`
- Then you can run `python -m src.data_prep` with only the `make_splits()` part if needed, but this script only prepares frames. It does NOT create splits.

Usage:
    python -m src.import_external_frames --src "D:/path/to/dataset_root"

Notes:
- Classes must match the project classes in `src/config.py`.
- If your dataset is already at 150x150 and named correctly, you can skip this.
"""
import os
import sys
import argparse
from glob import glob
import cv2

from src.config import CLASSES, FRAMES_PER_VIDEO, IMG_H, IMG_W, FRAMES_DIR


def _evenly_spaced_indices(n: int, k: int):
    if n <= 0 or k <= 0:
        return []
    return sorted({int(round(i)) for i in [j * (n - 1) / max(k - 1, 1) for j in range(k)]})


def import_from_src(src_root: str, overwrite: bool = False) -> None:
    src_root = os.path.abspath(src_root)
    if not os.path.isdir(src_root):
        raise FileNotFoundError(f"Source root not found: {src_root}")

    print(f"[INFO] Scanning external dataset at: {src_root}")
    total_videos = 0
    saved_images = 0

    for cls in CLASSES:
        cls_src = os.path.join(src_root, cls)
        if not os.path.isdir(cls_src):
            print(f"[WARN] Class folder missing in source: {cls_src}")
            continue
        # video_id assumed as subfolder name under cls
        video_dirs = [d for d in glob(os.path.join(cls_src, '*')) if os.path.isdir(d)]
        print(f"[INFO] Class '{cls}': {len(video_dirs)} video folders found")
        out_dir = os.path.join(FRAMES_DIR, cls)
        os.makedirs(out_dir, exist_ok=True)

        for vdir in sorted(video_dirs):
            vid = os.path.basename(vdir)
            # collect image frames
            frames = []
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
                frames.extend(glob(os.path.join(vdir, ext)))
            frames = sorted(frames)
            if len(frames) < FRAMES_PER_VIDEO:
                continue
            total_videos += 1

            idxs = _evenly_spaced_indices(len(frames), FRAMES_PER_VIDEO)
            for i, idx in enumerate(idxs):
                src_img = frames[idx]
                dst_img = os.path.join(out_dir, f"{vid}_f{i}.jpg")
                if os.path.exists(dst_img) and not overwrite:
                    continue
                img = cv2.imread(src_img)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
                cv2.imwrite(dst_img, img)
                saved_images += 1

    print(f"[INFO] Imported videos: {total_videos}")
    print(f"[INFO] Saved resized frames: {saved_images}")
    print(f"[INFO] Frames are in: {FRAMES_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='Path to external dataset root')
    ap.add_argument('--overwrite', action='store_true', help='Overwrite existing frames')
    args = ap.parse_args()
    import_from_src(args.src, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
