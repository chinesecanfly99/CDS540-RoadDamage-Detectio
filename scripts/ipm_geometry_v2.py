"""
Inverse Perspective Mapping (IPM) Geometry Module for Road Plane Area Estimation (v2).
Course: CDS540 Computer Vision (C-Tier Upgrade C1)

DISCLAIMER (R5 / D06):
Because RDD2022 does not include camera intrinsics, extrinsics, mounting height,
or lane width calibration, any metric area computed from monocular dashcam frames
is an ASSUMED-GEOMETRY ESTIMATE (`estimated_ground_area_m2`) with explicit
uncertainty, NOT a true physical ground-truth area.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def canonical_road_quad(image_shape: tuple[int, ...]) -> np.ndarray:
    """
    Conservative canonical forward dashcam road trapezoid (TL, TR, BR, BL)
    for uncalibrated frames when vanishing-point calibration is unavailable.
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    return np.array(
        [
            [0.36 * w, 0.52 * h],  # Top-Left (far left)
            [0.64 * w, 0.52 * h],  # Top-Right (far right)
            [0.92 * w, 0.94 * h],  # Bottom-Right (near right)
            [0.08 * w, 0.94 * h],  # Bottom-Left (near left)
        ],
        dtype=np.float32,
    )


class RoadPlaneIPM:
    """
    Homography-based road-plane Inverse Perspective Mapping with explicit
    geometric assumptions and out-of-region / horizon uncertainty handling.
    """

    def __init__(
        self,
        lane_width_m: float = 3.5,
        lane_length_m: float = 20.0,
        camera_height_m: float = 1.2,
        seed: int = 42,
        bev_size: tuple[int, int] = (360, 720),
    ) -> None:
        if lane_width_m <= 0 or lane_length_m <= 0 or camera_height_m <= 0:
            raise ValueError("lane_width_m, lane_length_m, and camera_height_m must be positive.")
        self.lane_width_m = float(lane_width_m)
        self.lane_length_m = float(lane_length_m)
        self.camera_height_m = float(camera_height_m)
        self.seed = int(seed)
        self.bev_width = int(bev_size[0])
        self.bev_height = int(bev_size[1])

        self.H: np.ndarray | None = None
        self.H_inv: np.ndarray | None = None
        self.src_quad: np.ndarray | None = None
        self.image_shape: tuple[int, int] | None = None
        self.calibration_mode: str = "uncalibrated"

    @property
    def dst_quad(self) -> np.ndarray:
        """Destination bird's-eye rectangle corners (TL, TR, BR, BL)."""
        w, h = float(self.bev_width), float(self.bev_height)
        return np.array(
            [
                [0.0, 0.0],
                [w - 1.0, 0.0],
                [w - 1.0, h - 1.0],
                [0.0, h - 1.0],
            ],
            dtype=np.float32,
        )

    @property
    def m2_per_bev_pixel(self) -> float:
        """Metric surface conversion derived directly from assumed lane rectangle."""
        bev_px_area = float(self.bev_width * self.bev_height)
        return (self.lane_width_m * self.lane_length_m) / bev_px_area

    def calibrate_from_quad(self, src_quad: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
        """
        Calibrate 3x3 homography H from a 4x2 float32 road trapezoid
        ordered as [top-left, top-right, bottom-right, bottom-left].
        """
        quad = np.asarray(src_quad, dtype=np.float32)
        if quad.shape != (4, 2):
            raise ValueError(f"src_quad must have shape (4, 2), got {quad.shape}")
        if not np.all(np.isfinite(quad)):
            raise ValueError("src_quad contains non-finite values.")

        H = cv2.getPerspectiveTransform(quad, self.dst_quad)
        if H is None or not np.all(np.isfinite(H)):
            raise RuntimeError("Failed to compute valid homography matrix from src_quad.")

        H = H.astype(np.float64)
        # Normalize homography sign so that the calibrated road trapezoid center has positive homogeneous w.
        quad_center = np.array([quad[:, 0].mean(), quad[:, 1].mean(), 1.0], dtype=np.float64)
        if float((H @ quad_center)[2]) < 0.0:
            H = -H

        self.H = H
        self.H_inv = np.linalg.inv(self.H)
        self.src_quad = quad.copy()
        self.image_shape = (int(image_shape[0]), int(image_shape[1]))
        self.calibration_mode = "quad"
        return self.H.copy()

    def calibrate_from_vanishing_point(
        self, image_bgr: np.ndarray
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """
        Attempt lane-edge Hough line detection and vanishing-point intersection.
        Must return (None, {"reason": ...}) on failure; never silently falls back.
        """
        if image_bgr is None or image_bgr.ndim != 3:
            return None, {"reason": "invalid_input_image"}

        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 160)

        # Restrict to lower road region (below 45% image height)
        mask = np.zeros_like(edges)
        roi_top = int(0.45 * h)
        mask[roi_top:, :] = 255
        roi_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(
            roi_edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=40,
            minLineLength=max(24, int(0.08 * w)),
            maxLineGap=25,
        )
        if lines is None or len(lines) < 2:
            return None, {"reason": "insufficient_hough_segments", "num_lines": 0 if lines is None else len(lines)}

        left_lines: list[tuple[float, float]] = []
        right_lines: list[tuple[float, float]] = []
        for seg in lines.reshape(-1, 4):
            x1, y1, x2, y2 = [float(v) for v in seg]
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < 1e-3:
                continue
            slope = dy / dx
            # Reject near-horizontal and near-vertical lines
            if abs(slope) < 0.28 or abs(slope) > 4.0:
                continue
            intercept = y1 - slope * x1
            # In image coords (y down): negative slope on left side, positive slope on right side
            xm = 0.5 * (x1 + x2)
            if slope < 0 and xm < 0.65 * w:
                left_lines.append((slope, intercept))
            elif slope > 0 and xm > 0.35 * w:
                right_lines.append((slope, intercept))

        if not left_lines or not right_lines:
            return None, {
                "reason": "missing_converging_lane_pair",
                "left_count": len(left_lines),
                "right_count": len(right_lines),
            }

        m_left = float(np.median([m for m, _ in left_lines]))
        b_left = float(np.median([b for _, b in left_lines]))
        m_right = float(np.median([m for m, _ in right_lines]))
        b_right = float(np.median([b for _, b in right_lines]))

        denom = m_left - m_right
        if abs(denom) < 1e-3:
            return None, {"reason": "parallel_lane_candidates"}

        vx = (b_right - b_left) / denom
        vy = m_left * vx + b_left

        # Sanity-check vanishing point location
        if not (0.15 * w <= vx <= 0.85 * w and 0.15 * h <= vy <= 0.68 * h):
            return None, {
                "reason": "vanishing_point_out_of_plausible_horizon",
                "vp_x": round(vx, 2),
                "vp_y": round(vy, 2),
            }

        y_top = max(vy + 0.08 * h, 0.50 * h)
        y_bot = 0.94 * h
        if y_bot - y_top < 0.15 * h:
            return None, {"reason": "degenerate_road_vertical_span", "y_top": round(y_top, 2), "y_bot": round(y_bot, 2)}

        x_tl = (y_top - b_left) / m_left
        x_tr = (y_top - b_right) / m_right
        x_bl = (y_bot - b_left) / m_left
        x_br = (y_bot - b_right) / m_right

        if not (0 <= x_tl < x_tr <= w and -0.25 * w <= x_bl < x_br <= 1.25 * w):
            return None, {
                "reason": "reconstructed_quad_invalid_geometry",
                "quad_x": [round(x_tl, 1), round(x_tr, 1), round(x_br, 1), round(x_bl, 1)],
            }

        quad = np.array([[x_tl, y_top], [x_tr, y_top], [x_br, y_bot], [x_bl, y_bot]], dtype=np.float32)
        H = self.calibrate_from_quad(quad, image_bgr.shape)
        self.calibration_mode = "vanishing_point"
        return H, {
            "reason": "ok",
            "vp_x": round(vx, 2),
            "vp_y": round(vy, 2),
            "left_count": len(left_lines),
            "right_count": len(right_lines),
        }

    def warp_image(self, image_bgr: np.ndarray) -> np.ndarray:
        """Warp the road image into the bird's-eye view rectangle."""
        if self.H is None:
            raise RuntimeError("RoadPlaneIPM must be calibrated before calling warp_image().")
        return cv2.warpPerspective(
            image_bgr,
            self.H,
            (self.bev_width, self.bev_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(20, 20, 20),
        )

    def _project_points(self, pts: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
        """Project Nx2 points through H while guarding against horizon singularity."""
        if self.H is None:
            raise RuntimeError("RoadPlaneIPM must be calibrated before projection.")
        pts_h = np.hstack([pts.astype(np.float64), np.ones((len(pts), 1), dtype=np.float64)])
        proj = (self.H @ pts_h.T).T
        w_coord = proj[:, 2]
        # Positive w ensures points lie on the ground side of the camera horizon
        if np.any(w_coord <= 1e-4):
            return None, w_coord
        xy = proj[:, :2] / w_coord[:, None]
        if not np.all(np.isfinite(xy)):
            return None, w_coord
        return xy, w_coord

    def box_to_ground_polygon(self, box_xyxy: Any) -> np.ndarray | None:
        """
        Project an image-space [x1, y1, x2, y2] bounding box onto the calibrated
        ground-plane bird's-eye coordinate system (4x2 float32 polygon), or None
        if the box falls outside the calibrated ground horizon / road zone.
        """
        if self.H is None or self.src_quad is None or self.image_shape is None:
            raise RuntimeError("RoadPlaneIPM is not calibrated.")

        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        if x2 <= x1 or y2 <= y1:
            return None

        # Top horizon line of the calibrated trapezoid
        y_horizon = float(min(self.src_quad[0, 1], self.src_quad[1, 1]))
        # If the box sits completely above the road horizon (sky / far background), reject
        if y2 <= y_horizon:
            return None

        # Clamp top edge slightly inside valid ground projection horizon to avoid singularity
        y1_clamped = max(y1, y_horizon - 0.04 * self.image_shape[0])
        if y2 <= y1_clamped:
            return None

        corners = np.array(
            [[x1, y1_clamped], [x2, y1_clamped], [x2, y2], [x1, y2]],
            dtype=np.float64,
        )
        warped, _ = self._project_points(corners)
        if warped is None:
            return None

        # Reject extreme projective blow-ups far outside the calibrated lane corridor
        margin_x = 0.75 * self.bev_width
        margin_y = 0.45 * self.bev_height
        if (
            np.min(warped[:, 0]) < -margin_x
            or np.max(warped[:, 0]) > self.bev_width + margin_x
            or np.min(warped[:, 1]) < -margin_y
            or np.max(warped[:, 1]) > self.bev_height + margin_y
        ):
            return None

        return warped.astype(np.float32)

    @staticmethod
    def _polygon_area(poly: np.ndarray) -> float:
        x = poly[:, 0].astype(np.float64)
        y = poly[:, 1].astype(np.float64)
        return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    def estimated_ground_area_m2(self, box_xyxy: Any) -> tuple[float, float]:
        """
        Estimate ground-plane footprint area in m^2 under assumed lane geometry.
        Returns (estimated_ground_area_m2, confidence in [0, 1]).
        Returns (nan, 0.0) when the box falls outside the calibrated ground zone.
        """
        poly = self.box_to_ground_polygon(box_xyxy)
        if poly is None:
            return float("nan"), 0.0

        bev_area_px = self._polygon_area(poly)
        if bev_area_px <= 0.0 or not math.isfinite(bev_area_px):
            return float("nan"), 0.0

        area_m2 = bev_area_px * self.m2_per_bev_pixel

        # Compute geometric confidence in [0.10, 0.95] based on:
        # (1) fraction of warped polygon inside the calibrated BEV rectangle
        # (2) near-field vs far-field vertical position (perspective stretch sensitivity)
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        poly_clamped = poly.copy()
        poly_clamped[:, 0] = np.clip(poly_clamped[:, 0], 0.0, float(self.bev_width))
        poly_clamped[:, 1] = np.clip(poly_clamped[:, 1], 0.0, float(self.bev_height))
        in_area = self._polygon_area(poly_clamped)
        in_ratio = min(1.0, max(0.0, in_area / max(bev_area_px, 1e-6)))

        h_img = float(self.image_shape[0])
        y_horizon = float(min(self.src_quad[0, 1], self.src_quad[1, 1]))
        y_bot = float(max(self.src_quad[2, 1], self.src_quad[3, 1]))
        y_mid = 0.5 * (max(y1, y_horizon) + min(y2, h_img))
        depth_factor = min(1.0, max(0.15, (y_mid - y_horizon) / max(1.0, y_bot - y_horizon)))

        # Uncalibrated camera assumption caps max confidence below 0.90
        mode_cap = 0.88 if self.calibration_mode == "vanishing_point" else 0.78
        conf = mode_cap * (0.4 + 0.6 * in_ratio) * (0.35 + 0.65 * depth_factor)
        conf = round(float(min(0.95, max(0.05, conf))), 4)
        if in_ratio < 0.05:
            return float("nan"), 0.0
        return float(area_m2), conf


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate RoadPlaneIPM geometry module.")
    parser.add_argument("--dry-run", action="store_true", help="Validate imports, geometry math, and exit.")
    parser.add_argument("--lane-width", type=float, default=3.5)
    parser.add_argument("--lane-length", type=float, default=20.0)
    parser.add_argument("--camera-height", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ipm = RoadPlaneIPM(
        lane_width_m=args.lane_width,
        lane_length_m=args.lane_length,
        camera_height_m=args.camera_height,
        seed=args.seed,
    )
    sample_shape = (600, 600, 3)
    quad = canonical_road_quad(sample_shape)
    H = ipm.calibrate_from_quad(quad, sample_shape)

    # Sanity check near vs far boxes of equal 2D pixel area (60x40 = 2400 px)
    near_area, near_conf = ipm.estimated_ground_area_m2([270, 500, 330, 540])
    far_area, far_conf = ipm.estimated_ground_area_m2([270, 320, 330, 360])
    sky_area, sky_conf = ipm.estimated_ground_area_m2([270, 40, 330, 80])

    summary = {
        "module": "ipm_geometry_v2",
        "dry_run": bool(args.dry_run),
        "root": str(ROOT),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "homography_det": round(float(np.linalg.det(H)), 6),
        "near_box_equal_px_area_m2": round(near_area, 4),
        "near_conf": near_conf,
        "far_box_equal_px_area_m2": round(far_area, 4),
        "far_conf": far_conf,
        "sky_box_rejected": math.isnan(sky_area) and sky_conf == 0.0,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
