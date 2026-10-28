#!/usr/bin/env python3
"""
Interactive Web Application for Manual Sample Inspection & Correction
Multi-Image Visual Search Benchmark

Supports:
1. Full visual inspection of all samples:
   - Ground Truth images with bbox overlay, zoomed crops, and extracted attributes
   - Non-GT images (Hard Distractors & Other images) across Pool N=5, N=10, N=20
2. Case 1: GT Group Error (3 subtypes + explanation):
   - 1.1 Attribute wrong (question OK): Select valid GT images to keep (or edit their attributes),
         or select "No valid GT image at all" (Không có ảnh GT nào đúng), then Save.
   - 1.2 Question wrong: Auto-suggests alternative questions (condition + target attribute) ranked
         by keeping the MAXIMUM number of valid GT images, plus manual question/attribute override.
   - 1.3 Both question wrong & attribute wrong: Combines 1.1 (edit GT attributes & select valid GT
         images) and 1.2 (auto-suggests & edits question simultaneously).
3. Case 2: Non-GT Images Error (Distractor / Other wrong):
   - Provides an interactive Replacement Pool of valid candidate images (Hard Distractors & Other
     images) so the reviewer can swap any bad non-GT image with a clean image across all pools.
"""

import os
import sys
import copy
import json
import random
import argparse
import tempfile
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.review_logic import (
    format_qa_deterministic,
    sanitize_enriched_data,
    get_class_attribute_Spec,
    STOP_ATTRIBUTE_VALUES,
)
from src.review_logic import (
    find_hard_distractors,
    find_other_images,
    build_nested_pools,
)

CONFIG_PATH = Path(os.environ.get("REVIEW_CONFIG", ROOT / "config.yaml")).expanduser().resolve()
with CONFIG_PATH.open(encoding="utf-8") as config_file:
    CFG = yaml.safe_load(config_file)
PATHS = CFG["paths"]
DATA_ROOT = Path(os.environ.get("REVIEW_DATA_ROOT", str(ROOT / "demo"))).expanduser().resolve()
OUTPUT_ROOT = Path(os.environ.get("REVIEW_OUTPUT_DIR", str(ROOT / "reviews"))).expanduser().resolve()


def data_path(value):
    """Resolve legacy data/... paths against a separately mounted dataset."""
    path = Path(value)
    if path.is_absolute():
        return path
    parts = path.parts[1:] if path.parts and path.parts[0] == "data" else path.parts
    candidate = DATA_ROOT.joinpath(*parts).resolve()
    if not candidate.is_relative_to(DATA_ROOT):
        raise ValueError(f"Dataset path escapes REVIEW_DATA_ROOT: {value}")
    return candidate


@asynccontextmanager
async def lifespan(app):
    init_data()
    yield


app = FastAPI(title="Multi-Image Visual Search – Sample Review & Repair Web", lifespan=lifespan)

ORIGINAL_SAMPLES_PATH = data_path(PATHS["output_json"])
REVIEWED_SAMPLES_PATH = OUTPUT_ROOT / "samples_reviewed.json"
REVIEW_STATE_PATH = OUTPUT_ROOT / "sample_review_state.json"
ENRICHED_PATH = data_path(PATHS["enriched_attributes_json"])
FILTERED_BBOXES_PATH = data_path(PATHS.get("filtered_bboxes_json", "data/intermediate/filtered_bboxes.json"))
URLS_JSON_PATH = data_path(PATHS["image_urls_json"])
HTML_UI_PATH = ROOT / "review_ui.html"

# Serve only configured image folders, never the dataset's output/intermediate JSON.
IMAGE_DIRS = PATHS.get("image_dirs", ["data/pilot_5000_images", "data/pilot_10000_images", "data/pilot_1000_images"])
if not IMAGE_DIRS:
    raise ValueError("Configure at least one image directory in paths.image_dirs")
for folder in dict.fromkeys(IMAGE_DIRS + [PATHS.get("crops_dir", "data/crops")]):
    rel = folder.removeprefix("data/").strip("/")
    if not folder.startswith("data/") or not rel or ".." in Path(rel).parts or Path(folder).is_absolute():
        raise ValueError("Image folders must be relative data/... paths within the dataset")
    directory = data_path(folder)
    if directory.is_dir():
        app.mount("/data/" + rel, StaticFiles(directory=str(directory)), name=rel.replace("/", "_"))

STATE_LOCK = threading.RLock()


def synchronized(func):
    """Serialize state access within the supported single server process."""
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        with STATE_LOCK:
            return func(*args, **kwargs)
    return wrapper

# Global in-memory caches
STATE = {
    "samples": [],
    "samples_by_id": {},
    "original_by_id": {},
    "reviews": {},
    "raw_bboxes": [],
    "clean_bboxes": [],
    "filtered_bboxes": [],
    "raw_img_bboxes": defaultdict(list),
    "clean_img_bboxes": defaultdict(list),
    "imgs_with_target_by_obj": defaultdict(set),
    "fb_obj_counts": Counter(),
    "all_image_ids": [],
    "image_local_paths": {},
}


def resolve_img_rel_path(img_id: str, hint_path: str = "") -> str:
    for folder in IMAGE_DIRS:
        cand = data_path(folder) / f"{img_id}.jpg"
        if cand.exists():
            rel = f"{folder}/{img_id}.jpg"
            STATE["image_local_paths"][img_id] = rel
            return "/" + rel
    if hint_path:
        p = data_path(hint_path.lstrip("/"))
        if p.exists():
            return "/" + hint_path.lstrip("/")
    if img_id in STATE["image_local_paths"]:
        cached = STATE["image_local_paths"][img_id].lstrip("/")
        if data_path(cached).exists():
            return "/" + cached
    return f"/{IMAGE_DIRS[0]}/{img_id}.jpg"


def init_data():
    # Lifespan may be entered again by tests or app hosts.
    for key in ["raw_img_bboxes", "clean_img_bboxes", "imgs_with_target_by_obj", "fb_obj_counts", "image_local_paths"]:
        STATE[key].clear()
    STATE["raw_bboxes"] = []
    STATE["filtered_bboxes"] = []
    print("Loading benchmark data and caches for Review Web...")
    if not ORIGINAL_SAMPLES_PATH.exists():
        raise RuntimeError(f"Missing {ORIGINAL_SAMPLES_PATH}. See README.md for dataset setup.")

    with open(ORIGINAL_SAMPLES_PATH, "r", encoding="utf-8") as f:
        orig_samples = json.load(f)
    STATE["original_by_id"] = {s["sample_id"]: copy.deepcopy(s) for s in orig_samples}

    existing_on_disk = set()
    for folder in IMAGE_DIRS:
        for p in data_path(folder).glob("*.jpg"):
            if p.stat().st_size > 0:
                existing_on_disk.add(p.stem)
                STATE["image_local_paths"][p.stem] = f"{folder}/{p.name}"

    if REVIEWED_SAMPLES_PATH.exists():
        try:
            with open(REVIEWED_SAMPLES_PATH, "r", encoding="utf-8") as f:
                reviewed_samples = json.load(f)
            rev_map = {s["sample_id"]: s for s in reviewed_samples}
            merged = []
            for s in orig_samples:
                sid = s["sample_id"]
                if sid in rev_map:
                    rs = rev_map[sid]
                    # Replace any non-existent pool image IDs with the fixed ones from orig_samples
                    if existing_on_disk:
                        orig_neg_ids = [
                            it["image_id"]
                            for it in s.get("pools", {}).get("N_20", {}).get("images", [])
                            if it.get("role") != "gold" and it["image_id"] in existing_on_disk
                        ]
                        rep_idx = 0
                        rep_map = {}
                        for p_key in ["N_5", "N_10", "N_20"]:
                            for it in rs.get("pools", {}).get(p_key, {}).get("images", []):
                                if it["image_id"] not in existing_on_disk and orig_neg_ids:
                                    if it["image_id"] not in rep_map:
                                        rep_map[it["image_id"]] = orig_neg_ids[rep_idx % len(orig_neg_ids)]
                                        rep_idx += 1
                                    it["image_id"] = rep_map[it["image_id"]]
                    merged.append(rs)
                else:
                    merged.append(copy.deepcopy(s))
            STATE["samples"] = merged
        except Exception:
            STATE["samples"] = copy.deepcopy(orig_samples)
    else:
        STATE["samples"] = copy.deepcopy(orig_samples)

    STATE["samples_by_id"] = {s["sample_id"]: s for s in STATE["samples"]}

    if REVIEW_STATE_PATH.exists():
        try:
            with open(REVIEW_STATE_PATH, "r", encoding="utf-8") as f:
                STATE["reviews"] = json.load(f)
        except Exception:
            STATE["reviews"] = {}
    else:
        STATE["reviews"] = {}

    if ENRICHED_PATH.exists():
        with open(ENRICHED_PATH, "r", encoding="utf-8") as f:
            STATE["raw_bboxes"] = json.load(f)
    STATE["clean_bboxes"] = sanitize_enriched_data(STATE["raw_bboxes"])

    if FILTERED_BBOXES_PATH.exists():
        try:
            with open(FILTERED_BBOXES_PATH, "r", encoding="utf-8") as f:
                STATE["filtered_bboxes"] = json.load(f)
        except Exception:
            STATE["filtered_bboxes"] = []

    for b in STATE["raw_bboxes"]:
        iid = b.get("image_id")
        obj = b.get("object")
        if iid:
            STATE["raw_img_bboxes"][iid].append(b)
        if iid and obj:
            STATE["imgs_with_target_by_obj"][obj].add(iid)

    for b in STATE["clean_bboxes"]:
        iid = b.get("image_id")
        if iid:
            STATE["clean_img_bboxes"][iid].append(b)

    for fb in STATE["filtered_bboxes"]:
        dname = fb.get("DisplayName")
        iid = fb.get("ImageID")
        if dname and iid:
            STATE["imgs_with_target_by_obj"][dname].add(iid)
            STATE["fb_obj_counts"][(iid, dname)] += 1

    if existing_on_disk:
        STATE["all_image_ids"] = sorted(existing_on_disk)
    elif URLS_JSON_PATH.exists():
        with open(URLS_JSON_PATH, "r", encoding="utf-8") as f:
            urls_data = json.load(f)
        STATE["all_image_ids"] = list(urls_data.keys())
    else:
        STATE["all_image_ids"] = list(STATE["raw_img_bboxes"].keys())

    print(f"Loaded {len(STATE['samples'])} samples, {len(STATE['raw_bboxes'])} enriched bboxes, {len(STATE['all_image_ids'])} pool images.")


def save_state_to_disk():
    REVIEWED_SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    for path, content in [(REVIEWED_SAMPLES_PATH, STATE["samples"]), (REVIEW_STATE_PATH, STATE["reviews"])]:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
                temp_path = Path(f.name)
                json.dump(content, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


def get_all_sample_gt_images() -> set:
    gt_ids = set()
    for s in STATE["samples"]:
        for gt in s.get("ground_truth", []):
            gt_ids.add(gt["image_id"])
    return gt_ids


def enrich_sample_for_view(sample: dict) -> dict:
    """Attach resolved image URLs, crop URLs, distractor bboxes, and review metadata for UI."""
    sid = sample["sample_id"]
    s_view = copy.deepcopy(sample)
    target_obj = s_view["query"]["target_object"]
    spec = get_class_attribute_Spec(target_obj) or {}
    s_view["schema_spec"] = {
        "stable_attributes": spec.get("stable_attributes", ["color", "shape", "material", "pattern"]),
        "allowed_values": spec.get("allowed_values", {}),
    }

    for gt in s_view.get("ground_truth", []):
        gt["img_url"] = resolve_img_rel_path(gt["image_id"], gt.get("local_path", ""))
        for b in gt.get("gt_bboxes", []):
            cp = b.get("crop_path", "")
            b["crop_url"] = ("/" + cp.lstrip("/")) if cp else ""

    # Collect unique non-GT images across N_5, N_10, N_20 with their pool memberships
    pool_membership = defaultdict(list)
    non_gt_map = {}
    for p_key, p_label in [("N_5", "N=5"), ("N_10", "N=10"), ("N_20", "N=20")]:
        p_obj = s_view.get("pools", {}).get(p_key, {})
        for item in p_obj.get("images", []):
            if item.get("role") == "gold":
                continue
            iid = item["image_id"]
            pool_membership[iid].append(p_label)
            if iid not in non_gt_map:
                obj_bboxes_in_img = [
                    {
                        "bbox_id": rb.get("bbox_id"),
                        "bbox_norm": rb.get("bbox_norm", []),
                        "crop_url": ("/" + rb["crop_path"].lstrip("/")) if rb.get("crop_path") else "",
                        "area_ratio": rb.get("area_ratio", 0.0),
                        "size_category": rb.get("size_category", ""),
                        "attributes": rb.get("attributes", {}),
                    }
                    for rb in STATE["raw_img_bboxes"].get(iid, [])
                    if rb.get("object") == target_obj
                ]
                non_gt_map[iid] = {
                    "image_id": iid,
                    "role": item.get("role", "other"),
                    "reason": item.get("reason", ""),
                    "pool_index": item.get("pool_index", 0),
                    "img_url": resolve_img_rel_path(iid),
                    "pools_in": [],
                    "target_obj_bboxes": obj_bboxes_in_img,
                }
    for iid, m_list in pool_membership.items():
        if iid in non_gt_map:
            non_gt_map[iid]["pools_in"] = m_list

    s_view["non_gt_images"] = list(non_gt_map.values())
    s_view["review"] = STATE["reviews"].get(sid, {
        "status": "pending",
        "has_gt_error": False,
        "gt_error_type": None,
        "gt_error_explanation": "",
        "no_valid_gt_images": False,
        "kept_gt_image_ids": [gt["image_id"] for gt in s_view.get("ground_truth", [])],
        "has_non_gt_error": False,
        "non_gt_explanation": "",
        "replaced_non_gt_history": [],
        "rejected_non_gt_ids": [],
        "updated_at": None,
    })
    return s_view


def compute_question_suggestions_for_sample(
    target_obj: str,
    gt_list: list,
    kept_image_ids: Optional[list] = None,
    current_cond_type: str = "",
    current_cond_val: str = "",
    current_target_type: str = "",
) -> list:
    """
    Mục 1.2 & 1.3: Đề xuất sửa lại câu hỏi sao cho giữ lại được NHIỀU GT image đúng nhất.
    Duyệt qua toàn bộ các attribute của các ảnh GT (hoặc tập ảnh GT được tick giữ lại),
    xếp hạng theo:
      1. Số lượng ảnh GT giữ lại được nhiều nhất (retained_count giảm dần)
      2. Số giá trị đáp án phân biệt hợp lý (ưu tiên >= 2 khi retained_count >= 2)
      3. Không trùng hệt câu hỏi cũ (hoặc đánh dấu nếu là câu hỏi hiện tại với đáp án cập nhật)
    """
    spec = get_class_attribute_Spec(target_obj)
    categories = spec["stable_attributes"] if spec else ["color", "shape", "material", "pattern"]

    if kept_image_ids is not None:
        kept_set = set(kept_image_ids)
        active_gt = [g for g in gt_list if g["image_id"] in kept_set]
    else:
        active_gt = list(gt_list)

    if not active_gt:
        return []

    candidates_map = {}

    possible_conds = set()
    for g in active_gt:
        for b in g.get("gt_bboxes", []):
            attrs = b.get("attributes") or {}
            for cat in categories:
                val = attrs.get(cat)
                if val and str(val).strip().lower() not in STOP_ATTRIBUTE_VALUES and str(val).strip().lower() != "null":
                    possible_conds.add((cat, str(val).strip().lower()))

    for cond_cat, cond_val in sorted(possible_conds):
        for tgt_cat in categories:
            if tgt_cat == cond_cat:
                continue
            retained_imgs = []
            target_vals = set()
            per_img_vals = {}
            for g in active_gt:
                iid = g["image_id"]
                matching_b = [
                    b for b in g.get("gt_bboxes", [])
                    if str((b.get("attributes") or {}).get(cond_cat) or "").strip().lower() == cond_val
                ]
                if not matching_b:
                    continue
                vals_in_img = [
                    str((b.get("attributes") or {}).get(tgt_cat) or "").strip().lower()
                    for b in matching_b
                ]
                if any((not v) or v in STOP_ATTRIBUTE_VALUES or v == "null" for v in vals_in_img):
                    continue
                retained_imgs.append(iid)
                target_vals.update(vals_in_img)
                per_img_vals[iid] = sorted(set(vals_in_img))

            if not retained_imgs or not target_vals:
                continue

            vals_sorted = sorted(target_vals)
            qa = format_qa_deterministic(target_obj, cond_cat, cond_val, tgt_cat, vals_sorted)
            is_same_query = (
                cond_cat == current_cond_type
                and cond_val == current_cond_val
                and tgt_cat == current_target_type
            )
            K_ret = len(retained_imgs)
            has_good_diversity = (len(vals_sorted) >= 2) if K_ret >= 2 else (len(vals_sorted) >= 1)

            candidates_map[(cond_cat, cond_val, tgt_cat)] = {
                "condition_attribute_type": cond_cat,
                "condition_attribute_value": cond_val,
                "target_attribute_type": tgt_cat,
                "question": qa["question"],
                "answer": {
                    "list": vals_sorted,
                    "split_comma": qa["answer_split_comma"],
                    "freeform": qa["answer_freeform"],
                },
                "retained_image_ids": retained_imgs,
                "retained_count": K_ret,
                "total_active_gt": len(active_gt),
                "per_image_values": per_img_vals,
                "is_current_query": is_same_query,
                "has_good_diversity": has_good_diversity,
            }

    suggestions = list(candidates_map.values())
    suggestions.sort(key=lambda x: (
        -x["retained_count"],
        0 if x["has_good_diversity"] else 1,
        1 if x["is_current_query"] else 0,
        -len(x["answer"]["list"]),
        x["condition_attribute_type"],
        x["condition_attribute_value"],
        x["target_attribute_type"],
    ))
    return suggestions[:25]


@app.get("/", response_class=HTMLResponse)
def serve_index():
    if not HTML_UI_PATH.exists():
        return HTMLResponse("<h1>Missing review_ui.html</h1>", status_code=500)
    return HTMLResponse(HTML_UI_PATH.read_text(encoding="utf-8"))


@app.get("/api/samples")
@synchronized
def api_list_samples():
    summary = []
    counts = Counter()
    for s in STATE["samples"]:
        sid = s["sample_id"]
        rev = STATE["reviews"].get(sid, {})
        st = rev.get("status", "pending")
        counts[st] += 1
        summary.append({
            "sample_id": sid,
            "target_object": s["query"]["target_object"],
            "condition_type": s["query"]["condition_attribute"]["type"],
            "condition_value": s["query"]["condition_attribute"]["value"],
            "target_attribute_type": s["query"]["target_attribute_type"],
            "question": s["query"]["question"],
            "K": s["K_gt_images"],
            "status": st,
            "has_gt_error": rev.get("has_gt_error", False),
            "gt_error_type": rev.get("gt_error_type"),
            "no_valid_gt_images": rev.get("no_valid_gt_images", False),
            "has_non_gt_error": rev.get("has_non_gt_error", False),
        })
    return {
        "total": len(summary),
        "status_counts": dict(counts),
        "samples": summary,
    }


@app.get("/api/sample/{sample_id}")
@synchronized
def api_get_sample(sample_id: str):
    sample = STATE["samples_by_id"].get(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    view = enrich_sample_for_view(sample)
    view["question_suggestions"] = compute_question_suggestions_for_sample(
        target_obj=view["query"]["target_object"],
        gt_list=view["ground_truth"],
        kept_image_ids=view["review"].get("kept_gt_image_ids"),
        current_cond_type=view["query"]["condition_attribute"]["type"],
        current_cond_val=view["query"]["condition_attribute"]["value"],
        current_target_type=view["query"]["target_attribute_type"],
    )
    return view


@app.post("/api/sample/{sample_id}/compute_suggestions")
@synchronized
def api_compute_suggestions(sample_id: str, payload: dict = Body(...)):
    sample = STATE["samples_by_id"].get(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    target_obj = sample["query"]["target_object"]
    gt_list = payload.get("ground_truth", sample["ground_truth"])
    kept_ids = payload.get("kept_gt_image_ids")
    cond_type = payload.get("condition_type", sample["query"]["condition_attribute"]["type"])
    cond_val = payload.get("condition_value", sample["query"]["condition_attribute"]["value"])
    tgt_type = payload.get("target_attribute_type", sample["query"]["target_attribute_type"])

    suggestions = compute_question_suggestions_for_sample(
        target_obj=target_obj,
        gt_list=gt_list,
        kept_image_ids=kept_ids,
        current_cond_type=cond_type,
        current_cond_val=cond_val,
        current_target_type=tgt_type,
    )
    return {"suggestions": suggestions}


@app.post("/api/format_qa")
@synchronized
def api_format_qa(payload: dict = Body(...)):
    obj = payload.get("target_object", "")
    cond_type = payload.get("condition_type", "")
    cond_val = payload.get("condition_value", "")
    tgt_type = payload.get("target_attribute_type", "")
    values = payload.get("values", [])
    qa = format_qa_deterministic(obj, cond_type, cond_val, tgt_type, values)
    return qa


@app.get("/api/sample/{sample_id}/replacement_pool")
@synchronized
def api_get_replacement_pool(
    sample_id: str,
    cond_type: Optional[str] = None,
    cond_val: Optional[str] = None,
):
    """
    Trả về Pool ảnh thay thế (Hard Distractors & Other Images) cho sample_id
    để người dùng chọn thay thế các ảnh ngoài GT bị sai.
    """
    sample = STATE["samples_by_id"].get(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    target_obj = sample["query"]["target_object"]
    c_type = cond_type or sample["query"]["condition_attribute"]["type"]
    c_val = cond_val or sample["query"]["condition_attribute"]["value"]

    rev = STATE["reviews"].get(sample_id, {})
    rejected_ids = set(rev.get("rejected_non_gt_ids", []))

    current_pool_ids = set()
    for p_key in ["N_5", "N_10", "N_20"]:
        for it in sample.get("pools", {}).get(p_key, {}).get("images", []):
            current_pool_ids.add(it["image_id"])

    all_gt_ids = get_all_sample_gt_images()
    excluded = all_gt_ids | current_pool_ids | rejected_ids

    rng = random.Random(42)
    hard_cands = find_hard_distractors(
        target_object=target_obj,
        cond_cat=c_type,
        cond_val=c_val,
        excluded_img_ids=excluded,
        all_bboxes=STATE["raw_bboxes"],
        filtered_bboxes=STATE["filtered_bboxes"],
        num_needed=40,
        rng=rng,
        clean_img_bboxes=STATE["clean_img_bboxes"],
        raw_img_bboxes=STATE["raw_img_bboxes"],
        fb_obj_counts=STATE["fb_obj_counts"],
        imgs_with_target_by_obj=STATE["imgs_with_target_by_obj"],
    )

    hard_ids = {h["image_id"] for h in hard_cands}
    other_cands_ids = find_other_images(
        target_object=target_obj,
        excluded_img_ids=excluded | hard_ids,
        all_image_ids=STATE["all_image_ids"],
        all_bboxes=STATE["raw_bboxes"],
        filtered_bboxes=STATE["filtered_bboxes"],
        num_needed=40,
        rng=rng,
        imgs_with_target_by_obj=STATE["imgs_with_target_by_obj"],
    )

    hard_list = []
    for h in hard_cands:
        iid = h["image_id"]
        obj_bboxes = [
            {
                "bbox_id": rb.get("bbox_id"),
                "bbox_norm": rb.get("bbox_norm", []),
                "crop_url": ("/" + rb["crop_path"].lstrip("/")) if rb.get("crop_path") else "",
                "area_ratio": rb.get("area_ratio", 0.0),
                "size_category": rb.get("size_category", ""),
                "attributes": rb.get("attributes", {}),
            }
            for rb in STATE["raw_img_bboxes"].get(iid, [])
            if rb.get("object") == target_obj
        ]
        hard_list.append({
            "image_id": iid,
            "role": "hard_distractor",
            "reason": h["reason"],
            "img_url": resolve_img_rel_path(iid),
            "target_obj_bboxes": obj_bboxes,
        })

    other_list = [
        {
            "image_id": oid,
            "role": "other",
            "reason": f"Negative image containing no {target_obj}",
            "img_url": resolve_img_rel_path(oid),
            "target_obj_bboxes": [],
        }
        for oid in other_cands_ids
    ]

    return {
        "sample_id": sample_id,
        "target_object": target_obj,
        "condition_attribute": {"type": c_type, "value": c_val},
        "hard_distractor_candidates": hard_list,
        "other_image_candidates": other_list,
    }


@app.post("/api/sample/{sample_id}/replace_non_gt")
@synchronized
def api_replace_non_gt(sample_id: str, payload: dict = Body(...)):
    """
    Thay thế 1 ảnh ngoài GT bị sai (`old_image_id`) bằng 1 ảnh mới (`new_image_id`) từ Pool
    trong toàn bộ các cấu hình N=5, N=10, N=20 và pools_by_K.
    """
    sample = STATE["samples_by_id"].get(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    old_iid = payload.get("old_image_id")
    new_iid = payload.get("new_image_id")
    new_role = payload.get("new_role", "other")
    new_reason = payload.get("new_reason", "Manually selected replacement image")
    note = payload.get("note", "")

    if not old_iid or not new_iid:
        raise HTTPException(status_code=400, detail="old_image_id and new_image_id are required")

    def replace_in_pools_dict(pools_dict: dict):
        for n_key, p_data in (pools_dict or {}).items():
            for item in p_data.get("images", []):
                if item.get("role") != "gold" and item.get("image_id") == old_iid:
                    item["image_id"] = new_iid
                    item["role"] = new_role
                    item["reason"] = new_reason
            p_data["num_hard_distractors"] = sum(
                1 for i in p_data.get("images", []) if i.get("role") == "hard_distractor"
            )
            p_data["num_other_images"] = sum(
                1 for i in p_data.get("images", []) if i.get("role") == "other"
            )

    replace_in_pools_dict(sample.get("pools", {}))
    for k_key, p_by_k in (sample.get("pools_by_K") or {}).items():
        replace_in_pools_dict(p_by_k)

    rev = STATE["reviews"].setdefault(sample_id, {
        "status": "pending",
        "has_gt_error": False,
        "gt_error_type": None,
        "gt_error_explanation": "",
        "no_valid_gt_images": False,
        "kept_gt_image_ids": [gt["image_id"] for gt in sample.get("ground_truth", [])],
        "has_non_gt_error": True,
        "non_gt_explanation": "",
        "replaced_non_gt_history": [],
        "rejected_non_gt_ids": [],
        "updated_at": None,
    })
    rev["has_non_gt_error"] = True
    if old_iid not in rev.setdefault("rejected_non_gt_ids", []):
        rev["rejected_non_gt_ids"].append(old_iid)
    rev.setdefault("replaced_non_gt_history", []).append({
        "old_image_id": old_iid,
        "new_image_id": new_iid,
        "new_role": new_role,
        "new_reason": new_reason,
        "note": note,
        "replaced_at": datetime.now().isoformat(),
    })
    if rev.get("status") == "pending":
        rev["status"] = "fixed"
    rev["updated_at"] = datetime.now().isoformat()

    save_state_to_disk()
    return api_get_sample(sample_id)


def rebuild_pools_preserving_existing_negatives(
    sample: dict,
    new_gold_ids: list,
    cond_cat: str,
    cond_val: str,
    rejected_non_gt_ids: set,
):
    """
    Khi số lượng ảnh GT thay đổi (ví dụ giảm từ K=4 xuống K'=2 hoặc 3 do bỏ ảnh GT sai),
    giữ nguyên các ảnh negative đã có trong pool N=20 và bù thêm đúng số ảnh negative còn thiếu
    để đảm bảo N=5 có đủ 5 ảnh, N=10 có đủ 10 ảnh, N=20 có đủ 20 ảnh.
    """
    target_obj = sample["query"]["target_object"]
    existing_negatives = []
    seen_neg = set()
    for p_key in ["N_5", "N_10", "N_20"]:
        for it in sample.get("pools", {}).get(p_key, {}).get("images", []):
            if it.get("role") != "gold":
                iid = it["image_id"]
                if iid not in seen_neg and iid not in rejected_non_gt_ids and iid not in new_gold_ids:
                    seen_neg.add(iid)
                    existing_negatives.append(it)

    needed_total_neg = 20 - len(new_gold_ids)
    if len(existing_negatives) < needed_total_neg:
        extra_needed = needed_total_neg - len(existing_negatives) + 5
        excluded = get_all_sample_gt_images() | seen_neg | set(new_gold_ids) | rejected_non_gt_ids
        rng = random.Random(12345)
        extra_hard = find_hard_distractors(
            target_object=target_obj,
            cond_cat=cond_cat,
            cond_val=cond_val,
            excluded_img_ids=excluded,
            all_bboxes=STATE["raw_bboxes"],
            filtered_bboxes=STATE["filtered_bboxes"],
            num_needed=extra_needed,
            rng=rng,
            clean_img_bboxes=STATE["clean_img_bboxes"],
            raw_img_bboxes=STATE["raw_img_bboxes"],
            fb_obj_counts=STATE["fb_obj_counts"],
            imgs_with_target_by_obj=STATE["imgs_with_target_by_obj"],
        )
        extra_hard_ids = {h["image_id"] for h in extra_hard}
        extra_other = find_other_images(
            target_object=target_obj,
            excluded_img_ids=excluded | extra_hard_ids,
            all_image_ids=STATE["all_image_ids"],
            all_bboxes=STATE["raw_bboxes"],
            filtered_bboxes=STATE["filtered_bboxes"],
            num_needed=extra_needed,
            rng=rng,
            imgs_with_target_by_obj=STATE["imgs_with_target_by_obj"],
        )
        for h in extra_hard:
            existing_negatives.append({
                "image_id": h["image_id"],
                "role": "hard_distractor",
                "reason": h["reason"],
            })
        for oid in extra_other:
            existing_negatives.append({
                "image_id": oid,
                "role": "other",
                "reason": f"Negative image containing no {target_obj}",
            })

    hard_list = [{"image_id": x["image_id"], "reason": x.get("reason", "")} for x in existing_negatives if x.get("role") == "hard_distractor"]
    other_list = [x["image_id"] for x in existing_negatives if x.get("role") != "hard_distractor"]
    hard_ratio = CFG["qa_generation"].get("negative_hard_ratio", 0.70)

    sample["pools"] = build_nested_pools(
        sample_gold_ids=new_gold_ids,
        available_gold_ids=new_gold_ids,
        hard_distractors=hard_list,
        other_img_ids=other_list,
        hard_ratio=hard_ratio,
        sample_rng=random.Random(42),
        n_configs=[5, 10, 20],
    )


@app.post("/api/sample/{sample_id}/save")
@synchronized
def api_save_sample_review(sample_id: str, payload: dict = Body(...)):
    """
    Lưu toàn bộ kết quả kiểm tra & chỉnh sửa của 1 sample:
    - verdict: 'pass' | 'fixed' | 'no_valid_gt'
    - Trường hợp 1 (GT group sai):
        - gt_error_type: 'attribute_wrong' | 'question_wrong' | 'both_wrong' | None
        - gt_error_explanation: string
        - no_valid_gt_images: bool (1.1 Không có ảnh GT nào đúng luôn)
        - kept_gt_image_ids: danh sách image_id của các ảnh GT đúng được giữ lại
        - updated_ground_truth: danh sách ground_truth (kèm các attribute đã chỉnh sửa nếu có)
        - updated_query: {question, condition_type, condition_value, target_attribute_type}
        - updated_answer: {list, split_comma, freeform}
    - Trường hợp 2 (Ảnh ngoài GT sai):
        - has_non_gt_error: bool
        - non_gt_explanation: string
    """
    sample = STATE["samples_by_id"].get(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    has_gt_error = bool(payload.get("has_gt_error", False))
    gt_error_type = payload.get("gt_error_type") if has_gt_error else None
    gt_error_explanation = (payload.get("gt_error_explanation") or "").strip()
    no_valid_gt = bool(payload.get("no_valid_gt_images", False))

    has_non_gt_error = bool(payload.get("has_non_gt_error", False))
    non_gt_explanation = (payload.get("non_gt_explanation") or "").strip()

    # Apply GT attribute edits if provided
    updated_gt_input = payload.get("updated_ground_truth")
    if isinstance(updated_gt_input, list) and updated_gt_input:
        gt_by_id = {g["image_id"]: g for g in sample.get("ground_truth", [])}
        for ug in updated_gt_input:
            iid = ug.get("image_id")
            if iid in gt_by_id and "gt_bboxes" in ug:
                for idx_b, ub in enumerate(ug["gt_bboxes"]):
                    if idx_b < len(gt_by_id[iid]["gt_bboxes"]) and "attributes" in ub:
                        gt_by_id[iid]["gt_bboxes"][idx_b]["attributes"] = ub["attributes"]

    kept_ids = payload.get("kept_gt_image_ids")
    if no_valid_gt:
        kept_ids = []
        status = "no_valid_gt"
    else:
        if kept_ids is None:
            kept_ids = [g["image_id"] for g in sample.get("ground_truth", [])]
        if len(kept_ids) == 0:
            no_valid_gt = True
            status = "no_valid_gt"
        else:
            status = payload.get("status") or ("fixed" if (has_gt_error or has_non_gt_error) else "pass")

    prev_rev = STATE["reviews"].get(sample_id, {})
    rejected_non_gt_ids = set(prev_rev.get("rejected_non_gt_ids", []))

    if not no_valid_gt and kept_ids:
        kept_set = set(kept_ids)
        new_gt_list = [g for g in sample.get("ground_truth", []) if g["image_id"] in kept_set]
        if new_gt_list:
            sample["ground_truth"] = new_gt_list
            sample["K_gt_images"] = len(new_gt_list)

            # Update query if changed
            uq = payload.get("updated_query") or {}
            target_obj = sample["query"]["target_object"]
            cond_type = uq.get("condition_type") or sample["query"]["condition_attribute"]["type"]
            cond_val = uq.get("condition_value") or sample["query"]["condition_attribute"]["value"]
            tgt_type = uq.get("target_attribute_type") or sample["query"]["target_attribute_type"]

            sample["query"]["condition_attribute"] = {"type": cond_type, "value": cond_val}
            sample["query"]["target_attribute_type"] = tgt_type

            # Compute values from kept GT images for tgt_type
            computed_vals = sorted({
                str((b.get("attributes") or {}).get(tgt_type) or "").strip().lower()
                for g in new_gt_list
                for b in g.get("gt_bboxes", [])
                if (b.get("attributes") or {}).get(tgt_type)
            } - {"", "null", "none"})

            ua = payload.get("updated_answer") or {}
            if ua.get("list"):
                ans_list = [str(x).strip() for x in ua["list"] if str(x).strip()]
            elif computed_vals:
                ans_list = computed_vals
            else:
                ans_list = sample["answer"]["list"]

            auto_qa = format_qa_deterministic(target_obj, cond_type, cond_val, tgt_type, ans_list)
            custom_q = (uq.get("question") or "").strip()
            sample["query"]["question"] = custom_q if custom_q else auto_qa["question"]
            sample["answer"] = {
                "list": ans_list,
                "split_comma": (ua.get("split_comma") or "").strip() or auto_qa["answer_split_comma"],
                "freeform": (ua.get("freeform") or "").strip() or auto_qa["answer_freeform"],
            }

            new_gold_ids = [g["image_id"] for g in new_gt_list]
            rebuild_pools_preserving_existing_negatives(
                sample=sample,
                new_gold_ids=new_gold_ids,
                cond_cat=cond_type,
                cond_val=cond_val,
                rejected_non_gt_ids=rejected_non_gt_ids,
            )

            # Update ground_truth_progression
            gt_prog = {}
            for k_val in range(1, len(new_gt_list) + 1):
                sub_gt = new_gt_list[:k_val]
                sub_ids = [g["image_id"] for g in sub_gt]
                sub_vals = sorted({
                    str((b.get("attributes") or {}).get(tgt_type) or "").strip().lower()
                    for g in sub_gt
                    for b in g.get("gt_bboxes", [])
                    if (b.get("attributes") or {}).get(tgt_type)
                } - {"", "null", "none"}) or ans_list
                sub_qa = format_qa_deterministic(target_obj, cond_type, cond_val, tgt_type, sub_vals)
                gt_prog[f"K_{k_val}"] = {
                    "K": k_val,
                    "gold_image_ids": sub_ids,
                    "answer": {
                        "list": sub_vals,
                        "split_comma": sub_qa["answer_split_comma"],
                        "freeform": sub_qa["answer_freeform"],
                    },
                }
            sample["ground_truth_progression"] = gt_prog

    STATE["reviews"][sample_id] = {
        "status": status,
        "has_gt_error": has_gt_error,
        "gt_error_type": gt_error_type,
        "gt_error_explanation": gt_error_explanation,
        "no_valid_gt_images": no_valid_gt,
        "kept_gt_image_ids": kept_ids,
        "has_non_gt_error": has_non_gt_error or bool(prev_rev.get("replaced_non_gt_history")),
        "non_gt_explanation": non_gt_explanation,
        "replaced_non_gt_history": prev_rev.get("replaced_non_gt_history", []),
        "rejected_non_gt_ids": list(rejected_non_gt_ids),
        "updated_at": datetime.now().isoformat(),
    }

    save_state_to_disk()
    return api_get_sample(sample_id)


@app.post("/api/sample/{sample_id}/reset")
@synchronized
def api_reset_sample(sample_id: str):
    """Khôi phục sample về trạng thái gốc ban đầu."""
    if sample_id not in STATE["original_by_id"]:
        raise HTTPException(status_code=404, detail="Sample not found")
    orig_copy = copy.deepcopy(STATE["original_by_id"][sample_id])
    for idx, s in enumerate(STATE["samples"]):
        if s["sample_id"] == sample_id:
            STATE["samples"][idx] = orig_copy
            break
    STATE["samples_by_id"][sample_id] = orig_copy
    STATE["reviews"].pop(sample_id, None)
    save_state_to_disk()
    return api_get_sample(sample_id)


@app.get("/api/export")
@synchronized
def api_export_verified():
    """Xuất file JSON tổng hợp kết quả review và các sample hợp lệ (loại bỏ các sample no_valid_gt)."""
    valid_samples = []
    for s in STATE["samples"]:
        sid = s["sample_id"]
        rev = STATE["reviews"].get(sid, {})
        if rev.get("no_valid_gt_images") or rev.get("status") == "no_valid_gt":
            continue
        s_out = copy.deepcopy(s)
        s_out["manual_review"] = rev
        valid_samples.append(s_out)

    export_payload = {
        "exported_at": datetime.now().isoformat(),
        "total_original_samples": len(STATE["samples"]),
        "total_valid_samples": len(valid_samples),
        "reviews_by_sample": STATE["reviews"],
        "samples": valid_samples,
    }
    return JSONResponse(content=export_payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Sample Review & Repair Web Server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
