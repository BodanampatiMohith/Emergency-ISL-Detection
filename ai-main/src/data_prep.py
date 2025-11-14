"""Data preparation utilities
- Extract 5 evenly-spaced frames per video from `data/raw/<class>/*.mp4`
- Resize to 150x150, normalize to [0,1] at model input (not saved normalized)
- Save frames as `data/processed/frames/<class>/<video_id>_f{i}.jpg`
- Create 60/20/20 train/val/test splits by video_id and write lists in `data/processed/splits/*.txt`
"""
import os
import glob
import cv2
import json
import random
from typing import List, Tuple
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_VIDEOS_DIR,
    FRAMES_DIR,
    SPLIT_DIR,
    CLASSES,
    CLASS_TO_ID,
    FRAMES_PER_VIDEO,
    IMG_H,
    IMG_W,
)


def _evenly_spaced_indices(num_frames: int, k: int) -> List[int]:
    if num_frames <= 0:
        return []
    if k <= 0:
        return []
    # Use linspace-like selection in integer index space
    return sorted({int(round(i)) for i in [j * (num_frames - 1) / max(k - 1, 1) for j in range(k)]})


def extract_frames_from_video(video_path: str, out_dir: str, k_frames: int = FRAMES_PER_VIDEO) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = _evenly_spaced_indices(total, k_frames)

    saved = []
    for i, frame_idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.resize(frame, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        fname = f"{os.path.splitext(os.path.basename(video_path))[0]}_f{i}.jpg"
        fpath = os.path.join(out_dir, fname)
        cv2.imwrite(fpath, frame)
        saved.append(fpath)

    cap.release()
    return saved


def prepare_frames() -> None:
    print(f"[INFO] Searching raw videos in: {RAW_VIDEOS_DIR}")
    for cls in CLASSES:
        cand_dirs = sorted(glob.glob(os.path.join(RAW_VIDEOS_DIR, f"{cls}*")))
        videos = []
        valid_ext = {".mp4", ".avi", ".mov", ".mkv"}
        for d in cand_dirs:
            if os.path.isdir(d):
                for root, _dirs, files in os.walk(d):
                    for fn in files:
                        ext = os.path.splitext(fn)[1].lower()
                        if ext in valid_ext:
                            videos.append(os.path.join(root, fn))
        out_dir = os.path.join(FRAMES_DIR, cls)
        os.makedirs(out_dir, exist_ok=True)
        print(f"[INFO] Class '{cls}': {len(videos)} videos")
        for vp in videos:
            extract_frames_from_video(vp, out_dir)


def _collect_video_ids() -> List[Tuple[str, str]]:
    # Return list of (class, video_id) from saved frame files
    pairs = []
    for cls in CLASSES:
        cdir = os.path.join(FRAMES_DIR, cls)
        frames = sorted(glob.glob(os.path.join(cdir, "*.jpg")))
        # group by video_id prefix before `_f{i}`
        vid_groups = {}
        for fp in frames:
            base = os.path.basename(fp)
            vid_id = base.split("_f")[0]
            vid_groups.setdefault(vid_id, 0)
            vid_groups[vid_id] += 1
        for vid_id, cnt in vid_groups.items():
            if cnt >= FRAMES_PER_VIDEO:
                pairs.append((cls, vid_id))
    return pairs


def make_splits(seed: int = 42) -> None:
    pairs = _collect_video_ids()
    random.seed(seed)
    pairs = sorted(pairs)
    if len(pairs) == 0:
        raise ValueError(
            "No videos/frames found. Please place videos under 'data/raw/<class>/' and rerun src.data_prep, "
            "or ask to generate a small synthetic dataset to proceed."
        )
    y = [cls for cls, _ in pairs]

    # stratify over class labels at the video level
    train_pairs, tmp_pairs = train_test_split(
        pairs, test_size=0.4, random_state=seed, stratify=y
    )
    y_tmp = [cls for cls, _ in tmp_pairs]
    val_pairs, test_pairs = train_test_split(
        tmp_pairs, test_size=0.5, random_state=seed, stratify=y_tmp
    )

    for name, subset in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        out_txt = os.path.join(SPLIT_DIR, f"{name}.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            for cls, vid in subset:
                f.write(json.dumps({"class": cls, "video_id": vid}) + "\n")
        print(f"[INFO] Wrote {name} split with {len(subset)} video IDs -> {out_txt}")


if __name__ == "__main__":
    prepare_frames()
    make_splits()
