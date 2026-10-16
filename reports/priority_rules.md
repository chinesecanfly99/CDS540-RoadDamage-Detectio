# Municipal Road Maintenance Prioritization Rules & Triage Specification

**Course**: Lingnan University CDS540 Computer Vision  
**Project**: Automated Road Damage Detection for Smart City Maintenance  
**Module**: Automated Decision Support & Operational Triage (`scripts/priority.py`)  
**Deliverable**: Step 11 — Tasks 11.1, 11.2, 11.3  

---

## 1. Operational Rationale & Municipal Stakeholder Context

In municipal road asset management, raw bounding box coordinates $[x_1, y_1, x_2, y_2]$ and class labels provide insufficient guidance for dispatching repair crews. Road maintenance budgets are constrained, and municipal authorities require an **automated risk-based triage score** that balances two distinct dimensions:
1. **Acute Safety & Liability Hazard**: Potholes (`D40`) pose immediate danger to motor vehicles (tire blowouts, suspension damage) and vulnerable road users (cyclists, motorcyclists), carrying substantial municipal liability.
2. **Structural Degradation Trajectory**: Alligator cracking (`D20`) signals deep structural failure of the road base. If left unaddressed before rainy seasons, water ingress accelerates subgrade liquefaction, converting fatigue cracks into massive potholes. Conversely, superficial longitudinal cracks (`D00`) require routine crack sealing rather than emergency dispatch.

To bridge the gap between computer vision predictions and actionable municipal workflows, we implemented a deterministic, transparent heuristic triage module in [scripts/priority.py](file:///c:/Users/liyangshang/OneDrive/Desktop/岭南课程/计算机视觉/小组作业/scripts/priority.py).

---

## 2. Mathematical Prioritization Formulation

The total frame triage score $S_{\text{total}}$ is computed as follows:

$$S_{\text{total}} = M_{\text{multiplicity}} \times \sum_{i=1}^{N} \left( W_{\text{class}}(c_i) \times P(c_i) \times \gamma_{\text{size}}(\alpha_i) \right)$$

where:
- $N$ is the number of detected bounding boxes with confidence score $P(c_i) \ge 0.25$.
- $c_i \in \{\text{D00}, \text{D10}, \text{D20}, \text{D40}\}$ is the predicted distress class.
- $W_{\text{class}}$ is the municipal hazard weighting factor.
- $P(c_i)$ is the model confidence probability $[0.25, 1.0]$.
- $\alpha_i = \frac{\text{Area}(\text{box}_i)}{W_{\text{img}} \times H_{\text{img}}}$ is the relative bounding box area ratio.
- $\gamma_{\text{size}}$ is the nonlinear spatial extent scaling penalty.
- $M_{\text{multiplicity}}$ is the defect clustering bonus factor.

### 2.1 Hazard Weighting Matrix ($W_{\text{class}}$)

| Class Code | Distress Category | Base Weight ($W_{\text{class}}$) | Engineering Rationale |
|---|---|---:|---|
| **`D40`** | **Pothole** | **10.0** | Extreme safety hazard, immediate puncture/accident risk. Asymmetric municipal liability cost. |
| **`D20`** | **Alligator Cracking** | **6.5** | High structural risk: indicates base course exhaustion and moisture infiltration vulnerability. |
| **`D10`** | **Transverse Cracking** | **3.5** | Moderate risk: thermal contraction joint distress, affects vehicle ride quality. |
| **`D00`** | **Longitudinal Cracking** | **2.5** | Low/Moderate risk: wheel-path fatigue, early-stage distress manageable via crack sealing. |

### 2.2 Nonlinear Spatial Scaling Penalty ($\gamma_{\text{size}}$)

Distress patches occupying a large portion of the camera field of view represent severe physical road deterioration:
$$\gamma_{\text{size}}(\alpha_i) = \begin{cases} 
2.00 & \text{if } \alpha_i \ge 10\% \text{ (Extensive regional failure)} \\
1.50 & \text{if } 4\% \le \alpha_i < 10\% \text{ (Moderate distress patch)} \\
1.20 & \text{if } 1\% \le \alpha_i < 4\% \text{ (Localized defect)} \\
1.00 & \text{if } \alpha_i < 1\% \text{ (Minor / superficial hairline defect)}
\end{cases}$$

### 2.3 Compounding Multiplicity Bonus ($M_{\text{multiplicity}}$)

Co-occurring defects on a single roadway segment exacerbate driver evasion risks and indicate systemic pavement breakdown:
$$M_{\text{multiplicity}} = \begin{cases} 
1.40 & \text{if } N \ge 4 \text{ defects (Heavy defect clustering)} \\
1.20 & \text{if } 2 \le N < 4 \text{ defects (Multiple concurrent distresses)} \\
1.00 & \text{if } N \le 1 \text{ defect (Isolated distress)}
\end{cases}$$

### 2.4 Priority Level Classification & Dispatch Action

| Priority Tier | Score Range / Trigger | HUD Badge Color | Actionable Municipal Maintenance Order |
|---|---|---|---|
| **HIGH** | $S_{\text{total}} \ge 12.0$ **OR** Severe Pothole Trigger ($P(\text{D40}) \ge 0.35$ and $\alpha \ge 0.5\%$) | **Bright Red** (`#DC0000`) | **Emergency Crew Dispatch (24–48 hours)**: Deploy cold-patch asphalt filler and erect hazard warnings to avoid vehicle damage claims. |
| **MEDIUM** | $6.0 \le S_{\text{total}} < 12.0$ | **Amber / Orange** (`#FFA500`) | **Scheduled Resurfacing Queue (30 days)**: Schedule for municipal milling and asphalt overlay program in current quarter. |
| **LOW** | $0.1 \le S_{\text{total}} < 6.0$ | **Green** (`#00C800`) | **Preventive Maintenance**: Seal cracks with hot-pour bituminous sealant to prevent moisture penetration; re-inspect in 6 months. |
| **NONE** | $S_{\text{total}} = 0.0$ | **Gray** (`#B4B4B4`) | **Sound Pavement**: Pavement intact. Maintain standard biennial surveillance cycle. |

---

## 3. Empirical Test Results on Sample Images (Task 11.2)

We validated the heuristic engine across representative inspection images from the test set (`results/priority_tests/priority_evaluation_summary.csv`).

**Table 1: Test Results Across Representative Severity Scenarios**

| Image ID | Ground Truth Annotations | Detected Distress Count | Potholes (D40) | Alligator (D20) | Transverse (D10) | Longitudinal (D00) | Triage Score ($S_{\text{total}}$) | Assigned Priority | Operational Action Recommendation |
|---|---|---:|---:|---:|---:|---:|---:|:---:|---|
| `India_000061.jpg` | D40 (Pothole) | 1 | 1 | 0 | 0 | 0 | **4.49** | **HIGH** *(Trigger)* | Urgent repair crew dispatch (24-48h). Pothole safety warning. |
| `India_000209.jpg` | D40, D20 | 5 | 4 | 1 | 0 | 0 | **34.13** | **HIGH** | Urgent repair crew dispatch (24-48h). High cluster density. |
| `India_000302.jpg` | D20 (Alligator) | 1 | 0 | 1 | 0 | 0 | **7.33** | **MEDIUM** | Scheduled maintenance queue (30 days). Mill-and-overlay program. |
| `Czech_003338.jpg` | D00 (Longitudinal) | 1 | 0 | 0 | 0 | 1 | **2.31** | **LOW** | Preventive maintenance. Crack sealing to inhibit water ingress. |
| `Czech_003025.jpg` | None (Sound Road) | 0 | 0 | 0 | 0 | 0 | **0.00** | **NONE** | Pavement sound. Maintain standard biennial inspection schedule. |
| `India_000168.jpg` | D40 (Potholes) | 2 | 2 | 0 | 0 | 0 | **12.68** | **HIGH** | Urgent repair crew dispatch (24-48h). Cold-patch deployment. |

All annotated inspection cards displaying the Heads-Up Display (HUD) telemetry overlay have been rendered and archived under `results/priority_tests/`.

---

## 4. Engineering Limitations & Academic Disclaimers (Task 11.3)

In accordance with municipal engineering best practices, we explicitly document the following operational limitations:

1. **Proxy Nature of 2D Image Bounding Boxes**:
   Bounding box area $[(x_2 - x_1) \times (y_2 - y_1)]$ is measured in 2D pixel space and is subject to **perspective foreshortening**. A defect near the vehicle hood occupies vastly more pixels than a larger defect 30 meters ahead. Pixel area serves strictly as an operational proxy and cannot directly determine volumetric asphalt displacement ($m^3$).
2. **Absence of 3D Pothole Depth Data**:
   Standard monocular RGB dashcams cannot measure the vertical depth ($z$-axis) of a depression. A shallow puddle or surface gouge may produce similar 2D contours to a 10 cm axle-breaking pothole.
3. **Non-Geotechnical Certification**:
   The prioritization formula is a heuristic automated triage filter designed to prioritize visual review queues for municipal road departments. It does not replace structural pavement testing (e.g., Falling Weight Deflectometer or laser profilometers).
4. **Recommendation for Human-in-the-Loop Oversight**:
   All HIGH priority alerts flagged by the system should trigger a high-resolution snapshot review by a certified municipal road supervisor prior to dispatching heavy paving machinery.
