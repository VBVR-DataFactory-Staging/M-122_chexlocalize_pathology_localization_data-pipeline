"""M-122: CheXlocalize pathology localization — REAL rendering.

Loads real CheXpert val chest X-rays + decodes CheXlocalize COCO-RLE pathology
masks, produces animated videos:
  - first_frame.png  : raw CXR
  - final_frame.png  : CXR + red pathology overlay + label text
  - first_video.mp4  : 60 frames, CXR fade-in (black -> full brightness)
  - last_video.mp4   : 60 frames, CXR -> mask gradually revealed (alpha ramp)
  - ground_truth.mp4 : 96 frames, full walkthrough (fade-in + reveal + hold)
  - prompt.txt / metadata.json

Data locations on EC2 (synced by bootstrap):
  _extracted/M-122_CheXlocalize/chexlocalize/CheXlocalize/gt_segmentations_val.json
  _extracted/M-122_CheXlocalize/chexlocalize/CheXlocalize/gt_segmentations_test.json
  _extracted/M-122_CheXlocalize/chexlocalize/CheXpert/val/<patient>/<study>/view1_frontal.jpg
  _extracted/M-122_CheXlocalize/chexlocalize/CheXpert/test/<patient>/<study>/view1_frontal.jpg
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(line_buffering=True)

import json
from pathlib import Path
import cv2
import numpy as np
from common import DATA_ROOT, write_task, COLORS, fit_square

PID = "M-122"
TASK_NAME = "chexlocalize_pathology_localization"
FPS = 24
N_FIRST_FRAMES = 60      # 2.5s fade-in
N_LAST_FRAMES = 60       # 2.5s reveal
N_GT_FRAMES = 96         # 4s full walkthrough
OUT_SIZE = 512
MAX_SAMPLES = 800        # cap per run (187 val + 499 test pairs + multi-pathology each)

PROMPT = (
    "This is a frontal chest X-ray from the CheXpert validation set. "
    "A radiologist has annotated the region most consistent with the labelled "
    "pathology. Study the initial X-ray and the final annotated overlay: "
    "identify the pathology and describe the anatomic location of the highlighted "
    "region (e.g. left lower lobe, right hilum, retrocardiac)."
)


def rle_decode(counts, shape):
    """Decode COCO-format RLE mask. `shape` is [height, width] per CheXlocalize."""
    try:
        from pycocotools import mask as maskUtils
        rle = {
            "size": [int(shape[0]), int(shape[1])],
            "counts": counts.encode() if isinstance(counts, str) else counts,
        }
        m = maskUtils.decode(rle)
        return m.astype(np.uint8)
    except Exception as e:
        print(f"    rle_decode failed: {e}", flush=True)
        return np.zeros((int(shape[0]), int(shape[1])), dtype=np.uint8)


def resolve_cxr_path(patient_key: str, split: str) -> Path | None:
    """Map 'patient64622_study1_view1_frontal' + split -> image path."""
    # pattern: patient<ID>_study<N>_view<N>_<orient>
    parts = patient_key.split("_")
    if len(parts) < 4:
        return None
    patient, study, view, orient = parts[0], parts[1], parts[2], "_".join(parts[3:])
    root = DATA_ROOT / "_extracted" / "M-122_CheXlocalize" / "chexlocalize" / "CheXpert" / split
    p = root / patient / study / f"{view}_{orient}.jpg"
    return p if p.exists() else None


def load_cxr(img_path: Path) -> np.ndarray | None:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # Normalize contrast (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def draw_label(img: np.ndarray, pathology: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (OUT_SIZE, 44), (0, 0, 0), -1)
    cv2.putText(out, f"Pathology: {pathology}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["yellow"], 2, cv2.LINE_AA)
    return out


def make_overlay(cxr_sq: np.ndarray, mask_sq: np.ndarray,
                 alpha: float, pathology: str,
                 contour: bool = True) -> np.ndarray:
    """Overlay red mask at given alpha (0=raw, 1=full overlay)."""
    if alpha <= 0.0:
        return draw_label(cxr_sq, pathology)
    out = cxr_sq.copy()
    m_bool = mask_sq > 0
    if m_bool.any():
        layer = out.copy()
        layer[m_bool] = COLORS["red"]
        out = cv2.addWeighted(layer, float(alpha) * 0.6, out, 1.0 - float(alpha) * 0.6, 0)
        if contour and alpha >= 0.5:
            m8 = (m_bool.astype(np.uint8)) * 255
            cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, COLORS["red"], 2)
    return draw_label(out, pathology)


def fade_in_frames(cxr_sq: np.ndarray, n: int, pathology: str) -> list[np.ndarray]:
    """Black -> CXR fade in, with label bar appearing halfway."""
    frames = []
    for k in range(n):
        t = k / max(1, n - 1)
        img = (cxr_sq.astype(np.float32) * t).clip(0, 255).astype(np.uint8)
        if t > 0.5:
            img = draw_label(img, pathology)
        frames.append(img)
    return frames


def reveal_frames(cxr_sq: np.ndarray, mask_sq: np.ndarray, n: int,
                  pathology: str) -> list[np.ndarray]:
    """CXR only -> gradual mask reveal."""
    frames = []
    for k in range(n):
        t = k / max(1, n - 1)
        frames.append(make_overlay(cxr_sq, mask_sq, t, pathology, contour=True))
    return frames


def gt_walkthrough(cxr_sq: np.ndarray, mask_sq: np.ndarray, n: int,
                   pathology: str) -> list[np.ndarray]:
    """Full sequence: fade-in (0..0.3) -> hold CXR (0.3..0.45) -> reveal (0.45..0.85) -> hold (0.85..1.0)."""
    frames = []
    for k in range(n):
        t = k / max(1, n - 1)
        if t < 0.3:
            a = t / 0.3
            img = (cxr_sq.astype(np.float32) * a).clip(0, 255).astype(np.uint8)
            frames.append(draw_label(img, pathology) if a > 0.5 else img)
        elif t < 0.45:
            frames.append(draw_label(cxr_sq, pathology))
        elif t < 0.85:
            a = (t - 0.45) / 0.4
            frames.append(make_overlay(cxr_sq, mask_sq, a, pathology))
        else:
            frames.append(make_overlay(cxr_sq, mask_sq, 1.0, pathology))
    return frames


def process_one(patient_key: str, split: str, pathology: str, rle: dict,
                task_idx: int) -> bool:
    shape = rle.get("size", [2320, 2828])
    counts = rle.get("counts", "")

    cxr_path = resolve_cxr_path(patient_key, split)
    if cxr_path is None:
        print(f"    skip {patient_key} [{split}]: image not found", flush=True)
        return False

    cxr = load_cxr(cxr_path)
    if cxr is None:
        print(f"    skip {patient_key}: cxr decode failed", flush=True)
        return False

    mask = rle_decode(counts, shape)
    if mask.sum() < 50:
        # empty / near-empty mask (no pathology labelled here)
        print(f"    skip {patient_key}/{pathology}: empty mask", flush=True)
        return False

    # Resize cxr + mask (shape is HxW per CheXlocalize) to 512x512 with padding.
    # NOTE: CheXlocalize shape=[H, W] but pycocotools decodes column-major — mask
    # arrives with shape (H, W). CXR jpeg likewise (H, W, 3). Align via fit_square.
    mh, mw = mask.shape[:2]
    ch, cw = cxr.shape[:2]
    # If dims differ (they shouldn't for aligned CheXpert/CheXlocalize), resize mask to cxr.
    if (mh, mw) != (ch, cw):
        mask = cv2.resize(mask, (cw, ch), interpolation=cv2.INTER_NEAREST)

    cxr_sq = fit_square(cxr, OUT_SIZE)
    mask_sq = fit_square(mask, OUT_SIZE, is_mask=True)

    first_frame = draw_label(cxr_sq, pathology)
    final_frame = make_overlay(cxr_sq, mask_sq, 1.0, pathology)

    first_video = fade_in_frames(cxr_sq, N_FIRST_FRAMES, pathology)
    last_video = reveal_frames(cxr_sq, mask_sq, N_LAST_FRAMES, pathology)
    gt_video = gt_walkthrough(cxr_sq, mask_sq, N_GT_FRAMES, pathology)

    meta = {
        "task": "CheXlocalize pathology localization",
        "dataset": "CheXlocalize (Stanford AIMI) on CheXpert",
        "case_id": f"{patient_key}_{pathology.replace(' ', '_')}",
        "modality": "Chest X-ray (frontal)",
        "pathology": pathology,
        "split": split,
        "source_image": str(cxr_path.relative_to(DATA_ROOT)) if cxr_path.is_relative_to(DATA_ROOT) else cxr_path.name,
        "mask_area_px": int(mask_sq.sum()),
        "fps": FPS,
        "first_video_frames": N_FIRST_FRAMES,
        "last_video_frames": N_LAST_FRAMES,
        "gt_video_frames": N_GT_FRAMES,
        "case_type": "E_cxr_pathology_localization",
    }

    write_task(PID, TASK_NAME, task_idx, first_frame, final_frame,
               first_video, last_video, gt_video, PROMPT, meta, FPS)
    return True


def iterate_entries():
    """Yield (patient_key, split, pathology, rle_dict). Val first (smaller, faster)."""
    for split, fname in [("val", "gt_segmentations_val.json"),
                         ("test", "gt_segmentations_test.json")]:
        jpath = DATA_ROOT / "_extracted" / "M-122_CheXlocalize" / "chexlocalize" / "CheXlocalize" / fname
        if not jpath.exists():
            print(f"  MISSING: {jpath}", flush=True)
            continue
        data = json.loads(jpath.read_text())
        print(f"  {split}: {len(data)} patients in {fname}", flush=True)
        for patient_key, pathologies in data.items():
            for patho_name, rle in pathologies.items():
                if not isinstance(rle, dict):
                    continue
                yield patient_key, split, patho_name, rle


def main():
    print(f"[M-122] CheXlocalize pathology localization — real rendering", flush=True)
    print(f"[M-122] DATA_ROOT = {DATA_ROOT}", flush=True)
    written = 0
    skipped = 0
    for patient_key, split, pathology, rle in iterate_entries():
        if written >= MAX_SAMPLES:
            print(f"[M-122] reached MAX_SAMPLES={MAX_SAMPLES}, stopping", flush=True)
            break
        ok = process_one(patient_key, split, pathology, rle, written)
        if ok:
            written += 1
            if written % 25 == 0:
                print(f"[M-122] wrote {written} samples so far", flush=True)
        else:
            skipped += 1
    print(f"[M-122] DONE: {written} written, {skipped} skipped", flush=True)


if __name__ == "__main__":
    main()
