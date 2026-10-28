"""Review helpers extracted from the benchmark pipeline; no model dependencies."""

import random
from collections import defaultdict, Counter
from .attribute_schema import STOP_ATTRIBUTE_VALUES, get_class_attribute_Spec, build_attribute_taxonomy

IRREGULAR_PLURALS = {
    "person": "people",
    "man": "men",
    "woman": "women",
    "child": "children",
    "foot": "feet",
    "tooth": "teeth",
    "goose": "geese",
    "mouse": "mice",
    "sheep": "sheep",
    "fish": "fish",
    "deer": "deer",
    "aircraft": "aircraft",
    "knife": "knives",
    "leaf": "leaves",
    "scarf": "scarves",
    "shelf": "shelves",
    "wolf": "wolves",
    "life": "lives",
}

def pluralize(word: str) -> str:
    """Return English plural form for common objects."""
    w = word.strip().lower()
    if w in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[w]
    if w.endswith("s") or w.endswith("sh") or w.endswith("ch") or w.endswith("x") or w.endswith("z"):
        return f"{w}es"
    if w.endswith("y") and len(w) > 1 and w[-2] not in "aeiou":
        return f"{w[:-1]}ies"
    return f"{w}s"

def format_qa_deterministic(obj: str, cond_type: str, cond_val: str, target_attr: str, values: list) -> dict:
    """
    Generate natural question and answer adhering strictly to the Few-shot pattern in pilot.md.
    Supports all attribute categories in attribute_schema.py.
    """
    plural_obj = pluralize(obj)
    vals_clean = sorted(list(set(values)))
    comma_str = ", ".join(vals_clean)

    if len(vals_clean) == 1:
        conj_str = vals_clean[0]
    elif len(vals_clean) == 2:
        conj_str = f"{vals_clean[0]} and {vals_clean[1]}"
    elif len(vals_clean) > 2:
        conj_str = f"{', '.join(vals_clean[:-1])}, and {vals_clean[-1]}"
    else:
        conj_str = ""

    is_single = len(vals_clean) == 1
    attr_phrase = {
        "color": "color" if is_single else "colors",
        "shape": "shape" if is_single else "shapes",
        "material": "material" if is_single else "materials",
        "pattern": "pattern" if is_single else "patterns",
        "subtype": "type" if is_single else "types",
        "state": "state" if is_single else "states",
        "placement": "location" if is_single else "locations",
        "pose": "pose" if is_single else "poses",
        "orientation": "orientation" if is_single else "orientations",
    }.get(target_attr, target_attr)

    if cond_val and cond_val != "none":
        cv = cond_val.strip()
        is_post_modifier = (
            cv.startswith(("on ", "in ", "at ", "under ", "by "))
            or cv in ("indoors", "outdoors", "on road", "in water", "in sky", "on table")
        )
        if is_post_modifier:
            question = f"What {attr_phrase} are the {plural_obj} {cv}?"
            freeform = f"The {plural_obj} {cv} across the images are {conj_str}."
        else:
            question = f"What {attr_phrase} are {cv} {plural_obj}?"
            freeform = f"The {cv} {plural_obj} across the images are {conj_str}."
    else:
        question = f"What {attr_phrase} are the {plural_obj}?"
        freeform = f"The {plural_obj} across the images are {conj_str}."

    return {
        "question": question,
        "answer_split_comma": comma_str,
        "answer_freeform": freeform,
    }

VALID_TAXONOMY = build_attribute_taxonomy()

MATERIAL_CLEANUP = {
    "woven": "fabric",
    "woven material": "fabric",
    "woven synthetic": "fabric",
    "synthetic fabric": "fabric",
    "cloth": "fabric",
    "textile": "fabric",
    "cotton": "fabric",
    "wood": "wooden",
    "timber": "wooden",
    "iron": "metal",
    "steel": "metal",
    "aluminum": "metal",
    "metallic": "metal",
    "porcelain": "ceramic",
    "clay": "ceramic",
    "transparent": "glass",
    "paper": "paper/cardboard",
    "cardboard": "paper/cardboard",
}

def sanitize_enriched_data(raw_data: list) -> list:
    """
    Filter high-confidence items và validate attribute theo spec của TỪNG class
    trong attribute_schema.py. Không mutate dict gốc trong raw_data.
    """
    cleaned = []
    for r_orig in raw_data:
        if not r_orig.get("is_clear_object") or r_orig.get("confidence_score", 0.0) < 0.85:
            continue
        r = dict(r_orig)
        attrs = r.get("attributes", {}) or {}
        spec = get_class_attribute_Spec(r.get("object", ""))
        allowed_by_cat = spec["allowed_values"] if spec else VALID_TAXONOMY
        clean_attrs = {}
        for cat, v in attrs.items():
            if not v or cat not in allowed_by_cat:
                continue
            v = str(v).lower().strip()
            if cat == "material":
                v = MATERIAL_CLEANUP.get(v, v)
            if v in allowed_by_cat[cat] and v not in STOP_ATTRIBUTE_VALUES:
                clean_attrs[cat] = v
        r["attributes"] = clean_attrs
        cleaned.append(r)
    return cleaned

def find_hard_distractors(
    target_object: str,
    cond_cat: str,
    cond_val: str,
    excluded_img_ids: set,
    all_bboxes: list,
    filtered_bboxes: list,
    num_needed: int,
    rng: random.Random,
    clean_img_bboxes: dict = None,
    raw_img_bboxes: dict = None,
    fb_obj_counts: dict = None,
    imgs_with_target_by_obj: dict = None,
) -> list:
    """
    Find non-GT images not in excluded_img_ids that strictly satisfy:
    - NO object in the image satisfies the gold group (target_object, cond_cat == cond_val).
    - Ideal Hard Distractor: contains target_object, EVERY instance of target_object in the image
      (across clean, raw, and filtered_bboxes) has an explicit non-null attribute cond_cat != cond_val.
    - Fallback Hard Distractor: contains ZERO instances of target_object (across clean, raw, and
      filtered_bboxes), and contains cond_cat == cond_val on a different object class.
    """
    if clean_img_bboxes is None:
        clean_bboxes = sanitize_enriched_data(all_bboxes)
        clean_img_bboxes = defaultdict(list)
        for b in clean_bboxes:
            clean_img_bboxes[b["image_id"]].append(b)

    if raw_img_bboxes is None:
        raw_img_bboxes = defaultdict(list)
        for b in all_bboxes:
            raw_img_bboxes[b["image_id"]].append(b)

    if fb_obj_counts is None:
        fb_obj_counts = Counter()
        for fb in filtered_bboxes:
            if fb.get("ImageID") and fb.get("DisplayName"):
                fb_obj_counts[(fb["ImageID"], fb["DisplayName"])] += 1

    if imgs_with_target_by_obj is None:
        imgs_with_target_by_obj = defaultdict(set)
        for b in all_bboxes:
            if b.get("object"):
                imgs_with_target_by_obj[b["object"]].add(b["image_id"])
        for fb in filtered_bboxes:
            if fb.get("DisplayName") and fb.get("ImageID"):
                imgs_with_target_by_obj[fb["DisplayName"]].add(fb["ImageID"])

    imgs_with_any_target = imgs_with_target_by_obj.get(target_object, set())

    ideal_hard = []
    fallback_hard = []

    for img_id, c_list in clean_img_bboxes.items():
        if img_id in excluded_img_ids:
            continue

        clean_obj_in_img = [b for b in c_list if b["object"] == target_object]
        raw_obj_in_img = [b for b in raw_img_bboxes.get(img_id, []) if b.get("object") == target_object]
        fb_cnt = fb_obj_counts.get((img_id, target_object), 0)

        if clean_obj_in_img:
            # Strict check: ALL instances of target_object in raw and filtered_bboxes must be in clean_obj_in_img
            if len(clean_obj_in_img) != len(raw_obj_in_img) or len(clean_obj_in_img) < fb_cnt:
                continue
            # Every single instance of target_object must have a known non-null value for cond_cat that is != cond_val
            cond_vals_in_img = [b.get("attributes", {}).get(cond_cat) for b in clean_obj_in_img]
            if any((not v) or (v == cond_val) for v in cond_vals_in_img):
                continue

            other_vals = sorted(set(str(v) for v in cond_vals_in_img))
            reason = f"Contains {target_object} with {cond_cat}='{', '.join(other_vals)}' (!= '{cond_val}')"
            ideal_hard.append({"image_id": img_id, "reason": reason})
            continue

        # Fallback candidate: image has ZERO instances of target_object anywhere, and has cond_val on another class
        if img_id not in imgs_with_any_target and not raw_obj_in_img and fb_cnt == 0:
            other_objs_with_cond = [
                b for b in c_list
                if b["object"] != target_object and b.get("attributes", {}).get(cond_cat) == cond_val
            ]
            if other_objs_with_cond:
                other_obj_name = other_objs_with_cond[0]["object"]
                fallback_hard.append({
                    "image_id": img_id,
                    "reason": f"Contains {other_obj_name} with {cond_cat}='{cond_val}' (no {target_object} in image)"
                })

    rng.shuffle(ideal_hard)
    rng.shuffle(fallback_hard)

    selected = ideal_hard[:num_needed]
    if len(selected) < num_needed:
        needed_more = num_needed - len(selected)
        selected.extend(fallback_hard[:needed_more])

    return selected

def find_other_images(
    target_object: str,
    excluded_img_ids: set,
    all_image_ids: list,
    all_bboxes: list,
    filtered_bboxes: list,
    num_needed: int,
    rng: random.Random,
    imgs_with_target_by_obj: dict = None,
) -> list:
    """Find images from pilot pool that contain ZERO instances of target_object across all annotations."""
    if imgs_with_target_by_obj is not None:
        imgs_with_target = imgs_with_target_by_obj.get(target_object, set())
    else:
        imgs_with_target = set(b["image_id"] for b in all_bboxes if b.get("object") == target_object)
        for fb in filtered_bboxes:
            if fb.get("DisplayName") == target_object:
                imgs_with_target.add(fb["ImageID"])

    excluded = excluded_img_ids.union(imgs_with_target)
    candidates = [iid for iid in all_image_ids if iid not in excluded]
    rng.shuffle(candidates)
    return candidates[:num_needed]

def build_nested_pools(
    sample_gold_ids: list,
    available_gold_ids: list,
    hard_distractors: list,
    other_img_ids: list,
    hard_ratio: float,
    sample_rng: random.Random,
    n_configs: list = [5, 10, 20]
) -> dict:
    """
    Construct prefix-nested evaluation pools for N in {5, 10, 20}:
    - For N=5: K GT images + (5 - K) negative images.
    - For N=10: ALL 5 images from N=5 + EXACTLY 5 new negative images.
    - For N=20: ALL 10 images from N=10 + EXACTLY 10 new negative images.

    Guarantees:
    - Pool(N=10)[:5] == Pool(N=5)
    - Pool(N=20)[:10] == Pool(N=10)
    """
    K = len(sample_gold_ids)
    M_20 = 20 - K

    hard_items = [
        {"image_id": hd["image_id"], "role": "hard_distractor", "reason": hd["reason"]}
        for hd in hard_distractors
    ]
    other_items = [
        {"image_id": oid, "role": "other", "reason": "Negative image containing no target object"}
        for oid in other_img_ids
    ]

    # 1. Segment 1 for N=5: M5 = 5 - K negatives
    M_5 = 5 - K
    H_5 = min(len(hard_items), round(hard_ratio * M_5))
    seg1_hard = hard_items[:H_5]
    needed_other_5 = M_5 - len(seg1_hard)
    seg1_other = other_items[:needed_other_5]
    D_5 = seg1_hard + seg1_other
    sample_rng.shuffle(D_5)

    rem_hard_1 = hard_items[len(seg1_hard):]
    rem_other_1 = other_items[len(seg1_other):]

    # 2. Segment 2 for N=10: exactly 5 new negatives (M10 = 10 - K)
    M_10 = 10 - K
    H_10 = round(hard_ratio * M_10)
    dH_10 = min(len(rem_hard_1), max(0, H_10 - len(seg1_hard)))
    seg2_hard = rem_hard_1[:dH_10]
    needed_other_10 = 5 - len(seg2_hard)
    seg2_other = rem_other_1[:needed_other_10]
    delta_D_10 = seg2_hard + seg2_other
    sample_rng.shuffle(delta_D_10)

    rem_hard_2 = rem_hard_1[len(seg2_hard):]
    rem_other_2 = rem_other_1[len(seg2_other):]

    # 3. Segment 3 for N=20: exactly 10 new negatives (M20 = 20 - K)
    M_20_hard_target = round(hard_ratio * M_20)
    dH_20 = min(len(rem_hard_2), max(0, M_20_hard_target - (len(seg1_hard) + len(seg2_hard))))
    seg3_hard = rem_hard_2[:dH_20]
    needed_other_20 = 10 - len(seg3_hard)
    seg3_other = rem_other_2[:needed_other_20]
    delta_D_20 = seg3_hard + seg3_other
    sample_rng.shuffle(delta_D_20)

    gold_items = [
        {"image_id": gid, "role": "gold"}
        for gid in sample_gold_ids
    ]

    # Pool N=5: Shuffle gold items + D_5 negatives
    pool_5_items = [dict(item) for item in gold_items + D_5]
    sample_rng.shuffle(pool_5_items)
    for p_idx, item in enumerate(pool_5_items):
        item["pool_index"] = p_idx

    # Pool N=10: Copy exact 5 items from Pool N=5 + append delta_D_10
    pool_10_items = [dict(item) for item in pool_5_items]
    for p_idx, item in enumerate(delta_D_10, start=5):
        it = dict(item)
        it["pool_index"] = p_idx
        pool_10_items.append(it)

    # Pool N=20: Copy exact 10 items from Pool N=10 + append delta_D_20
    pool_20_items = [dict(item) for item in pool_10_items]
    for p_idx, item in enumerate(delta_D_20, start=10):
        it = dict(item)
        it["pool_index"] = p_idx
        pool_20_items.append(it)

    pools = {
        "N_5": {
            "N": 5,
            "K": K,
            "num_hard_distractors": len([i for i in pool_5_items if i["role"] == "hard_distractor"]),
            "num_other_images": len([i for i in pool_5_items if i["role"] == "other"]),
            "images": pool_5_items
        },
        "N_10": {
            "N": 10,
            "K": K,
            "num_hard_distractors": len([i for i in pool_10_items if i["role"] == "hard_distractor"]),
            "num_other_images": len([i for i in pool_10_items if i["role"] == "other"]),
            "images": pool_10_items
        },
        "N_20": {
            "N": 20,
            "K": K,
            "num_hard_distractors": len([i for i in pool_20_items if i["role"] == "hard_distractor"]),
            "num_other_images": len([i for i in pool_20_items if i["role"] == "other"]),
            "images": pool_20_items
        }
    }

    return pools
