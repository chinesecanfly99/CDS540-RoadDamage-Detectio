# Dataset Partitioning & Cross-Domain Holdout Policy

**Course**: CDS540 Computer Vision  
**Project**: Automated Road Damage Detection for Smart City Maintenance  
**Random Seed**: `42` (Fixed for 100% scientific reproducibility)  
**Total Ground Truth Images**: 15,340 images across Czech, India, and United States  

---

## 1. Split Strategy & Leakage Prevention Protocol

To adhere to the rigorous experimental standards of CDS540 and prevent any optimistic performance bias:
1. **Zero Data Leakage**: Splitting is executed strictly at the **image level** (rather than the bounding box level). Any image, along with all of its bounding boxes, belongs exclusively to exactly one split.
2. **Stratified Sampling**: The in-domain dataset (Czech + India) is partitioned into **Train (70%)**, **Validation (15%)**, and **In-Domain Test (15%)** using stratified grouping across both **Country of Origin** and **Damage Presence (Positive vs Negative background frames)**. This guarantees identical category and country representation across splits.
3. **Dedicated Cross-Domain Holdout**: An entire geographical region (**United States**, 4,805 images) is quarantined from all model training, validation, and hyperparameter tuning. It serves exclusively as the **Zero-Shot Domain Shift Benchmark** in Step 10 to evaluate cross-continental generalization.
4. **Preservation of Background Images**: A representative fraction (37%) of undamaged road background images is systematically retained in Train, Val, and Test to enforce true negative learning and suppress false alarms on pavement markings and shadows.

---

## 2. Partition Summary Statistics

| Partition | Total Images | Positive (Damaged) | Negative (Background) | Total Bounding Boxes | Czech Images | India Images | US Images |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | **7,375** | 3,400 (46.1%) | 3,975 (53.9%) | 6,952 | 1,980 | 5,395 | 0 |
| **Validation** | **1,581** | 729 (46.1%) | 852 (53.9%) | 1,498 | 425 | 1,156 | 0 |
| **In-Domain Test** | **1,579** | 728 (46.1%) | 851 (53.9%) | 1,498 | 424 | 1,155 | 0 |
| **Domain Test (Holdout)** | **4,805** | 4,805 (100.0%) | 0 (0.0%) | 11,014 | 0 | 0 | 4,805 |
| **Combined Total** | **15,340** | **9,662** | **5,678** | **20,962** | **2,829** | **7,706** | **4,805** |

---

## 3. Roles and Operational Guidelines

* **`train.csv` (7,375 images)**: Used exclusively for updating neural network weights during YOLO backpropagation.
* **`val.csv` (1,581 images)**: Used for real-time validation after each epoch, early stopping trigger, and model checkpoint selection (`best.pt`).
* **`test.csv` (1,579 images)**: Used for evaluating the primary in-domain benchmark metrics (mAP@50, mAP@50-95, Class-specific Precision/Recall) after model training is completely finished.
* **`domain_test.csv` (4,805 images)**: Evaluated in Step 10 to quantify operational degradation under realistic domain shifts (unseen highway pavement, different dashboard cameras, US road markings).

---

## 4. Manifest Export Paths

All generated manifests are stored under `data/manifests/`:
- `data/manifests/train.csv`
- `data/manifests/val.csv`
- `data/manifests/test.csv`
- `data/manifests/domain_test.csv`
