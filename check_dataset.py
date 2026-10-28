#!/usr/bin/env python3
"""Read-only check of sample image/crop paths before starting the web."""
import json

import review_web as web


def main():
    if not web.ORIGINAL_SAMPLES_PATH.is_file():
        print(f"Missing required samples: {web.ORIGINAL_SAMPLES_PATH}")
        return 1
    samples = json.loads(web.ORIGINAL_SAMPLES_PATH.read_text(encoding="utf-8"))
    paths = set()
    for sample in samples:
        for gt in sample.get("ground_truth", []):
            paths.add(web.resolve_img_rel_path(gt["image_id"], gt.get("local_path", "")))
            for bbox in gt.get("gt_bboxes", []):
                if bbox.get("crop_path"):
                    paths.add(bbox["crop_path"])
        pool_sets = [sample.get("pools", {})] + list((sample.get("pools_by_K") or {}).values())
        for pools in pool_sets:
            for pool in pools.values():
                for item in pool.get("images", []):
                    paths.add(web.resolve_img_rel_path(item["image_id"]))
    missing = sorted(path for path in paths if not web.data_path(path.lstrip("/")).is_file())
    print(f"Dataset: {web.DATA_ROOT}")
    print(f"Samples: {len(samples)}; referenced image/crop paths: {len(paths)}; missing: {len(missing)}")
    for path in missing[:10]:
        print(f"  Missing: {path}")
    if len(missing) > 10:
        print(f"  ... and {len(missing) - 10} more")
    for label, path in [("enriched attributes", web.ENRICHED_PATH),
                        ("filtered bboxes", web.FILTERED_BBOXES_PATH),
                        ("image index", web.URLS_JSON_PATH)]:
        print(f"{'OK' if path.is_file() else 'WARNING missing'} {label}: {path}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
