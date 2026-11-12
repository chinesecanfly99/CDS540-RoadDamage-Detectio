# CHANGELOG_C — CDS540 C-Tier Technical Upgrade Log

> **Target Spec**: `C档技术升级_修改方案_v1.md`
> **Completed**: 2026-11-12 (Revision v2 after Audit Report `C档_审核报告_v1.md`)
> **Scope**: C1 (IPM Geometry), C2 (P3/P4/P5 Layer Activation Energy), C3 (US Few-Shot Adaptation with Strict Holdout), C4 (ONNX Export, Parity & Host Benchmark), C5 (Candidate Docs & Discrepancy Log).
> **Redlines**: R1 (118 protected files immutable), R2 (`v2` namespace isolation), R3–R10 strictly verified.

---

## 1. Environment & Dependency Audit (R9)

Baseline verified environment (`./.venv/Scripts/python.exe`):
- `torch`: `2.14.1+cu130` (NVIDIA GeForce RTX 5070 Laptop GPU, Blackwell `sm_120`)
- `ultralytics`: `8.4.177`, `opencv-python`: `5.0.0.93`, `numpy`: `2.5.2`, `pandas`: `3.0.6`, `matplotlib`: `3.11.2`, `scipy`: `1.18.1`
- Zero existing packages downgraded or uninstalled.

Added dependencies (purely additive):
- `./.venv/Scripts/python.exe -m pip install onnx onnxruntime` -> `onnx==1.23.2`, `onnxruntime==1.31.1` (`CPUExecutionProvider` active), `protobuf==7.36.2`, `ml_dtypes==0.6.0`, `flatbuffers==25.12.19`
- Auto-installed during Ultralytics ONNX export (`simplify=True`): `onnxslim==0.1.99`

---

## 2. Design Decisions, Conservations & Negative Results

### C1 — IPM Reverse Perspective Mapping
- **Assumed Road Trapezoid**: Canonical `y in [0.52H, 0.94H]`, top `x in [0.36W, 0.64W]`, bottom `x in [0.08W, 0.92W]`, `lane_width_m=3.5, lane_length_m=20.0, camera_height_m=1.2`.
- **Homography Sign Normalization & OpenCV 5.0 Fix**: Normalized `H` sign so calibrated centroid has positive `w`; fixed `HoughLinesP` iteration via `lines.reshape(-1, 4)`.
- **Key Findings & Negative Results** (40 stratified test frames, 97 GT boxes: 91 in-zone, 6 horizon-rejected):
  - Box-level area correlation: Spearman `0.8957`, Kendall `0.7370`, Top-10 overlap 8/10 (`top10_rank_differences.csv` updated with `weights_version` and `eval_split` columns per R3).
  - Priority stability (`RoadDamagePrioritizerV2`): 22/40 frames had zero model detections (`NONE` tier). Priority tiers changed on **2/40 frames overall (5.0%)**, or **2/18 frames with detections (11.1%)**; triage score Spearman `0.9946` overall / **`0.9505` on the 18 detected frames**.
  - Lane width sensitivity (`3.0 / 3.5 / 4.0 m`): Absolute `estimated_ground_area_m2` scaled linearly (`5.50 / 6.42 / 7.33 m^2`, +/-14.3%) with 0 priority tier changes across the 18 detected frames.
  - Automated Hough vanishing-point calibration succeeded on only `9/40` frames (`22.5%`), failing cleanly with explicit `reason` on 31 frames.

### C2 — P3 / P4 / P5 Layer Activation Energy Analysis
- **Dynamic Layer Discovery**: `discover_pyramid_layers()` hooked `model.16` (`C3k2`, stride 8, 128ch), `model.19` (`C3k2`, stride 16, 256ch), `model.22` (`C3k2`, stride 32, 512ch) with zero hardcoded indices.
- **Metric & Sample Framing**: Per-channel RMS activation energy upsampled bilinearly and normalized by per-frame L2 norm across `25` stratified test frames (21 defect + 4 pure background controls, `58` CSV rows, 0 NaNs; `eval_split` corrected to `test_stratified_cam_25`).
- **Key Findings & Negative Results (Full Class Breakdown)**:
  - `BG_NEG` (`n=4`, detection rate `0%`): `P3/P4/P5` shares `34.41% / 33.34% / 32.25%`; energy `0.9873x / 0.9566x / 0.9255x` RMS.
  - `D10` (`n=6`, detection rate **`0%`**): `P3` share `36.05%` (slightly above background `34.41%`, `+1.64 pp`; `1.2427x` RMS), `P5` `29.03%` (`0.9906x` RMS).
  - `D00` (`n=12`, detection rate **`16.7%`**): `P3+P4` combined share `73.77%` (`33.69% / 40.08%`; `1.2008x / 1.4413x` RMS), `P5` `26.22%` (`0.9398x` RMS).
  - `D20` (`n=10`, detection rate **`50.0%`**): `P4/P5` shares `35.95% / 34.47%` (`1.3090x / 1.2493x` RMS).
  - `D40` (`n=26`, detection rate **`19.2%`**): `P3/P4/P5` shares `35.15% / 33.98% / 30.87%`; energy `1.3110x / 1.2730x / 1.1466x` RMS. Notably, normalized `P5` share (`30.87%`) fell below background (`32.25%`) even though absolute `P5` energy (`1.1466x`) remained elevated—indicating share-based vs. energy-based metrics diverge for `D40`.
  - Because measurements are on GT regions with low sample detection rates, results represent **correlative localization evidence (not proof of detection)** and are only **partially consistent** with motion-blur asymmetry (a class-matched D40 follow-up is needed).

### C3 — US Domain Few-Shot Adaptation (`seed=42`)
- **Strict Holdout Partitioning**: Split 4,805 labelled US frames into `fewshot_val` (`200`), nested `fewshot_train_{50,100,200}`, and `adapt_holdout` (`4,405`, 93 D40 images). `split_integrity.json` updated with explicit zero-overlap checks for `train50`, `train100`, `train200`, `val`, and `adapt_holdout`.
- **Key Findings & Negative Results**:
  - `adapt_holdout` (`4,405`): Re-measured baseline `mAP@50 = 0.0863` -> `50-shot`: `0.1504` -> `100-shot`: `0.1957` (+126.8%) -> `200-shot`: `0.1843`.
  - **Catastrophic Forgetting**: In-domain `test` (`1,579`) `mAP@50` fell from `0.2845` to `0.2579` (50-shot), `0.2198` (100-shot), `0.1420` (200-shot, `-50.1%`).
  - **D40 Negative Transfer**: US `D40_Recall` fell `18.35% -> 7.66%`; in-domain `D40_Recall` fell `39.74% -> 8.61%`.

### C4 — ONNX Export, Numerical Parity & Host Benchmark
- **Protocol & Provenance Fix**: `export_benchmark.csv` baseline row `eval_split` corrected to `baseline_runtime_csv (100 frames)` to match `scripts/benchmark_runtime.py` L204 (`30 warmup / 100 frames`), distinguishing it from the 200-frame ONNX benchmarks.
- **ASCII Temp-Dir Fix for INT8**: Quantized inside `tempfile.TemporaryDirectory()` to bypass `onnx.shape_inference.infer_shapes_path` non-ASCII path failure.
- **Key Findings**: `FP32` (`36.18 MB`, `PASS`, `0.0002 px`, `313.71 ms / 3.2 FPS`), `FP16` (`18.14 MB`, `FAIL` 1 dropped box, `0.7688 px`, `287.39 ms / 3.5 FPS`), `INT8_dynamic` (`9.41 MB`, `FAIL`, `51.45 px / 0.2186`, `303.07 ms / 3.3 FPS`).

### C5 — Candidate Documents & Audit Discrepancies
- Duplicate deliverables exist under `reports/` and `submission/`; `submission/` is authoritative for merging. `discrepancy_log.md` updated with corrected D-1 (`batch=8` for both train & val), corrected D-3 (end-to-end wall-clock vs CUDA Event pure-forward, both 100 frames), and new D-8 (`benchmark_runtime.py` docstring vs L204).

---

## 3. Chart & Visual Registry

### Primary Charts (R7: 300 DPI, English-Only, No Emoji)
- `11_ipm_birdseye_and_area_comparison.png` -> `results/v2/ipm/11_ipm_birdseye_and_area_comparison.png` (DONE, 300 DPI)
- `12_layer_cam_response.png` -> `results/v2/cam/12_layer_cam_response.png` (DONE, 300 DPI)
- `13_fewshot_domain_recovery.png` -> `results/v2/domain_adapt/13_fewshot_domain_recovery.png` (DONE, 300 DPI)
- `14_export_accuracy_speed.png` -> `results/v2/runtime/14_export_accuracy_speed.png` (RE-RENDERED per B6, 300 DPI, title updated to remove R10 literal substring)

### Auxiliary Diagnostic Visualizations (Non-Primary, Not Numbered 11–14)
- `results/v2/ipm/sample_cards/*.jpg` (8 side-by-side bird's-eye qualitative inspection cards)
- `results/v2/cam/sample_panels/*.png` (25 four-panel activation energy visualizations at **200 DPI** for qualitative per-frame inspection)

---

## 4. Pre-Merge Closure (M1–M3 per Audit Report v2)
- **M1 (`defense_qa_addendum.md`)**: Stripped all LaTeX macros (`$`, `\`) from Q2/Q7/Q8; compressed each primary `Model Answer` to ~98–102 words (~38–40s spoken budget) and moved granular breakdown metrics into `If pressed, add:` backup blocks.
- **M2 (`C_supplement_preview.pdf`)**: Compiled candidate supplement independently under exact ReportLab formatting (`Times-Roman 12pt`, `15pt` leading, `0.75 in` margins, A4) without touching `submission/CDS540_Final_Project_Report.pdf` (6 pages). Measured preview length: ~1.32 pages (2 pages, 983 chars on page 2), keeping combined page budget <= 8 pages.
- **M3 (Wide A3 Scope)**: A3 forbidden-phrase scan explicitly covers `results/v2/`, `submission/v2_candidates/`, `CHANGELOG_C.md`, and `scripts/*_v2.py` (excluding the intentional forbidden-term list definition in `domain_adapt_fewshot_v2.py` L7). In-figure chart text is verified via the generating Python script title strings; PNG binaries are not scanned by text grep.
