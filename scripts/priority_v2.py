"""
Road Maintenance Prioritization Heuristic Module v2 with Optional IPM Area Proxy.
Course: CDS540 Computer Vision (C-Tier Upgrade C1)

Extends `scripts/priority.py::RoadDamagePrioritizer` without modifying the original
module. When `use_ipm=False`, delegates directly to `RoadDamagePrioritizer` so
outputs match the v1 baseline field-for-field.

DISCLAIMER (R5 / D06):
Outputs remain an operational human-in-the-loop triage proxy. Estimated ground
area (`estimated_ground_area_m2`) relies on uncalibrated lane-geometry assumptions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ipm_geometry_v2 import RoadPlaneIPM, canonical_road_quad
from scripts.priority import (
    CLASS_FULL_NAMES,
    CLASS_MAP,
    CLASS_WEIGHTS,
    PRIORITY_LEVELS,
    RoadDamagePrioritizer,
)


class RoadDamagePrioritizerV2(RoadDamagePrioritizer):
    """
    v2 Prioritizer supporting optional Inverse Perspective Mapping (IPM) ground-area
    sizing (`use_ipm=True`). `use_ipm=False` guarantees identical output to v1.
    """

    def __init__(
        self,
        conf_threshold: float = 0.25,
        use_ipm: bool = False,
        ipm: RoadPlaneIPM | None = None,
        lane_width_m: float = 3.5,
        lane_length_m: float = 20.0,
        camera_height_m: float = 1.2,
    ) -> None:
        super().__init__(conf_threshold=conf_threshold)
        self.use_ipm = bool(use_ipm)
        self._custom_ipm = ipm
        self.lane_width_m = float(lane_width_m)
        self.lane_length_m = float(lane_length_m)
        self.camera_height_m = float(camera_height_m)

    def _get_calibrated_ipm(self, img_width: int, img_height: int) -> RoadPlaneIPM:
        if self._custom_ipm is not None:
            if self._custom_ipm.H is None:
                quad = canonical_road_quad((img_height, img_width))
                self._custom_ipm.calibrate_from_quad(quad, (img_height, img_width))
            return self._custom_ipm
        ipm = RoadPlaneIPM(
            lane_width_m=self.lane_width_m,
            lane_length_m=self.lane_length_m,
            camera_height_m=self.camera_height_m,
        )
        quad = canonical_road_quad((img_height, img_width))
        ipm.calibrate_from_quad(quad, (img_height, img_width))
        return ipm

    def evaluate_detections(
        self,
        boxes: Any,
        classes: Any,
        confidences: Any,
        img_width: int,
        img_height: int,
    ) -> dict[str, Any]:
        if not self.use_ipm:
            return super().evaluate_detections(boxes, classes, confidences, img_width, img_height)

        ipm = self._get_calibrated_ipm(int(img_width), int(img_height))
        assumed_corridor_m2 = ipm.lane_width_m * ipm.lane_length_m
        img_area = float(img_width * img_height)

        total_score = 0.0
        defect_summary = {"D00": 0, "D10": 0, "D20": 0, "D40": 0}
        detected_items: list[dict[str, Any]] = []
        has_severe_pothole = False

        for idx in range(len(boxes)):
            conf = float(confidences[idx])
            if conf < self.conf_threshold:
                continue

            cls_id = int(classes[idx])
            cls_code = CLASS_MAP.get(cls_id, "D00")
            defect_summary[cls_code] += 1

            x1, y1, x2, y2 = [float(v) for v in boxes[idx]]
            box_area = max(0.0, (x2 - x1) * (y2 - y1))
            area_ratio_2d = box_area / img_area

            est_m2, ipm_conf = ipm.estimated_ground_area_m2([x1, y1, x2, y2])
            if math.isfinite(est_m2) and ipm_conf > 0.0:
                effective_ratio = est_m2 / assumed_corridor_m2
                ipm_fallback = False
            else:
                effective_ratio = area_ratio_2d
                ipm_fallback = True

            base_wt = CLASS_WEIGHTS.get(cls_code, 2.0)
            if effective_ratio >= 0.10:
                size_factor = 2.0
            elif effective_ratio >= 0.04:
                size_factor = 1.5
            elif effective_ratio >= 0.01:
                size_factor = 1.2
            else:
                size_factor = 1.0

            item_score = base_wt * conf * size_factor
            total_score += item_score

            if cls_code == "D40" and conf >= 0.35 and effective_ratio >= 0.005:
                has_severe_pothole = True

            detected_items.append(
                {
                    "class_code": cls_code,
                    "class_name": CLASS_FULL_NAMES[cls_code],
                    "confidence": conf,
                    "area_ratio": area_ratio_2d,
                    "estimated_ground_area_m2": None if math.isnan(est_m2) else round(float(est_m2), 4),
                    "ipm_confidence": float(ipm_conf),
                    "ipm_fallback_to_2d": ipm_fallback,
                    "effective_area_ratio": round(float(effective_ratio), 6),
                    "item_score": item_score,
                    "box": [x1, y1, x2, y2],
                }
            )

        num_distress = len(detected_items)
        if num_distress >= 4:
            multiplicity_bonus = 1.4
        elif num_distress >= 2:
            multiplicity_bonus = 1.2
        else:
            multiplicity_bonus = 1.0

        final_score = round(total_score * multiplicity_bonus, 2)
        if has_severe_pothole or final_score >= PRIORITY_LEVELS["HIGH"]["min_score"]:
            level = "HIGH"
        elif final_score >= PRIORITY_LEVELS["MEDIUM"]["min_score"]:
            level = "MEDIUM"
        elif final_score >= PRIORITY_LEVELS["LOW"]["min_score"]:
            level = "LOW"
        else:
            level = "NONE"

        return {
            "total_score": final_score,
            "priority_level": level,
            "action_recommendation": PRIORITY_LEVELS[level]["action"],
            "total_distress_count": num_distress,
            "defect_summary": defect_summary,
            "items": detected_items,
            "has_critical_pothole": has_severe_pothole,
            "area_proxy_mode": "ipm_estimated_ground_area_m2",
            "ipm_assumptions": {
                "lane_width_m": ipm.lane_width_m,
                "lane_length_m": ipm.lane_length_m,
                "camera_height_m": ipm.camera_height_m,
            },
        }


def verify_v1_parity() -> dict[str, Any]:
    """Assert exact field-by-field parity between v1 and v2(use_ipm=False)."""
    v1 = RoadDamagePrioritizer(conf_threshold=0.25)
    v2_off = RoadDamagePrioritizerV2(conf_threshold=0.25, use_ipm=False)
    v2_on = RoadDamagePrioritizerV2(conf_threshold=0.25, use_ipm=True)

    rng = np.random.default_rng(42)
    test_cases = [
        (
            np.array([[270, 500, 330, 540], [260, 330, 310, 360], [100, 120, 160, 160]], dtype=float),
            np.array([3, 0, 1]),
            np.array([0.62, 0.45, 0.20]),  # 0.20 below conf_threshold
            600,
            600,
        ),
        (
            np.zeros((0, 4), dtype=float),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=float),
            720,
            720,
        ),
        (
            rng.uniform(50, 550, size=(5, 4)),
            np.array([0, 1, 2, 3, 2]),
            np.array([0.55, 0.42, 0.71, 0.30, 0.88]),
            640,
            640,
        ),
    ]

    for i, (boxes, classes, confs, w, h) in enumerate(test_cases):
        if len(boxes) > 0:
            boxes[:, 2] = np.maximum(boxes[:, 0] + 10, boxes[:, 2])
            boxes[:, 3] = np.maximum(boxes[:, 1] + 10, boxes[:, 3])
        r1 = v1.evaluate_detections(boxes, classes, confs, w, h)
        r2 = v2_off.evaluate_detections(boxes, classes, confs, w, h)
        assert r1 == r2, f"v1 vs v2(use_ipm=False) mismatch on case {i}: {r1} != {r2}"

    r_on = v2_on.evaluate_detections(test_cases[0][0], test_cases[0][1], test_cases[0][2], 600, 600)
    return {
        "v1_parity_use_ipm_false": "PASS (3 cases exact match)",
        "v2_use_ipm_true_sample_score": r_on["total_score"],
        "v2_use_ipm_true_sample_level": r_on["priority_level"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate RoadDamagePrioritizerV2 and v1 parity.")
    parser.add_argument("--dry-run", action="store_true", help="Run parity assertions and exit.")
    args = parser.parse_args()

    report = verify_v1_parity()
    report["dry_run"] = bool(args.dry_run)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

