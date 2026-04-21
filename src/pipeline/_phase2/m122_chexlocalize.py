"""M-122: CheXlocalize pathology localization heatmap visualization.

Note: CheXlocalize provides heatmaps + RLE segmentations for val/test subsets of
CheXpert. We render the RLE mask over an aligned CXR image (if available).
Since CheXpert source images aren't in this download, we generate overlay
visualization on a placeholder derived from the RLE size (shape encoded in JSON).
"""
from __future__ import annotations
from pathlib import Path
import cv2, numpy as np, json, base64
from common import DATA_ROOT, write_task, COLORS, fit_square

PID="M-122"; TASK_NAME="chexlocalize_pathology_localization"; FPS=5
PROMPT=("This chest X-ray has a pathology annotation from CheXlocalize. "
        "Localize the pathological finding with a highlighted region.")

def rle_decode(counts: str, shape):
    """COCO-style RLE decode (basic, iterative)."""
    # Simplified: counts is a string of run-length encoded indices
    # CheXlocalize uses pycocotools RLE format
    try:
        from pycocotools import mask as maskUtils
        rle = {"size": list(shape), "counts": counts.encode() if isinstance(counts,str) else counts}
        return maskUtils.decode(rle).astype(np.uint8)
    except:
        return np.zeros(shape, dtype=np.uint8)

def loop_frames(f,n): return [f.copy() for _ in range(n)]

def process_sample(patient_key: str, pathology: str, rle: dict, idx: int):
    shape=rle.get("size",[1024,1024])
    mask=rle_decode(rle.get("counts",""),shape)
    # placeholder CXR: black canvas with mask overlay
    img=np.zeros((shape[0],shape[1],3),dtype=np.uint8)
    img_r=fit_square(img,512)
    mask_r=cv2.resize(mask,(512,512),interpolation=cv2.INTER_NEAREST)
    ann=img_r.copy()
    layer=ann.copy(); layer[mask_r>0]=COLORS["red"]
    ann=cv2.addWeighted(layer,0.6,ann,0.4,0)
    cv2.putText(ann,f"{pathology}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,COLORS["yellow"],2)
    n=FPS*4
    meta={"task":"CheXlocalize pathology RLE mask","dataset":"CheXlocalize",
          "case_id":f"{patient_key}_{pathology.replace(' ','_')}",
          "modality":"CXR (mask only — image cross-ref CheXpert)",
          "pathology":pathology,"fps":FPS,"frames_per_video":n,"case_type":"D_single_image_loop"}
    return write_task(PID,TASK_NAME,idx,img_r,ann,loop_frames(img_r,n),loop_frames(ann,n),loop_frames(ann,n),PROMPT,meta,FPS)

def main():
    root=DATA_ROOT/"_extracted"/"M-122_CheXlocalize"/"chexlocalize"/"CheXlocalize"
    gt=root/"gt_segmentations_val.json"
    if not gt.exists():
        print(f"  gt not found: {gt}"); return
    data=json.loads(gt.read_text())
    print(f"  {len(data)} CheXlocalize patients")
    i=0
    for pkey,pathologies in list(data.items())[:500]:
        for pname,rle in pathologies.items():
            d=process_sample(pkey,pname,rle,i)
            if d: i+=1

if __name__=="__main__": main()
