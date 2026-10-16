# CDS540 Group Project: TA Topic Submission Message & Email Template
# 课题申报邮件与沟通模板

**Course**: Lingnan University CDS540 Computer Vision  
**Deadline**: October 16, 2026 (Topic Selection & Reporting to TA)  
**Recommended Attachment (建议附件)**: `submission/CDS540_Project_Topic_Proposal.pdf`  

---

## 📌 版本一：邮件 / Moodle 正式报题模板（中英双语版）

**邮件主题 / Subject**:  
`[CDS540 Group Project] Topic Selection & Team Charter Submission - Group [X] (Road Damage Detection)`  
*(如果尚未分配组号，可写为：`[CDS540 Group Project] Topic Selection Submission - Leader: Yangshang Li`)*  

**正文内容 / Body Text**:

```text
尊敬的助教 / Dear TA,

我们是 CDS540 小组（Group [X]），按照课程教学安排（10月16日课题申报截止节点），在此正式向您报送期末大作业课题选题与小组成员名单：

【课题类别 / Project Track】
Candidate Topic 3: Object Detection and Image Segmentation（对应选题推荐方向二：道路损坏检测）

【项目题目 / Project Title】
Automated Road Damage Detection & Operational Triage for Smart City Infrastructure Maintenance
（基于计算机视觉的智慧城市道路病害自动化检测与运维分级系统）

【小组成员信息 / Team Members】
- 组长 / Leader: 李杨尚 (Yangshang Li) | 学号: 5586669 | 邮箱: yangshangli@LN.hk
- 组员 1 / Member: 吴松芫 (Songyuan Wu) | 学号: 5550024 | 邮箱: songyuanwu@ln.hk
- 组员 2 / Member: 郭毅锦 (Yijin Guo) | 学号: 5585392 | 邮箱: yijinguo@ln.hk
- 组员 3 / Member: 宋婧仪 (Jingyi Song) | 学号: 5554757 | 邮箱: jingyisong3@ln.hk
- 组员 4 / Member: 黄浠沿 (Xiyan Huang) | 学号: 5594367 | 邮箱: xiyanhuang@ln.hk

【课题简介与核心亮点 / Project Highlights】
1. 实际痛点与市政责任非对称成本：
   传统道路巡检依赖人工徒步或昂贵的激光检测车（单车 >30万美元），难以常态化高频覆盖。本项目致力于将低成本公交车/环卫车行车记录仪转化为道路普查网络。同时针对市政道路养护中“漏检深坑洼（D40）会直接导致爆胎翻车和高额法律索赔”的非对称成本特点，将坑洼召回率（Pothole Recall）作为核心优化指标。
2. 数据工程规范性：
   采用权威 IEEE CRDDC RDD2022 数据集（涵盖捷克、印度、美国三地 19,208 张图像）。严格采用图像级固定随机种子（seed=42）无泄漏划分（70% Train, 15% Val, 15% Test），测试集保留 851 张纯净无破损道路背景帧以防虚警；并将 100% 美国道路数据（4,805 张图像）完全隔离作为跨国域（沥青 vs 混凝土路面）的零样本迁移泛化测试。
3. 双模型对比设计：
   对比超轻量基线 YOLO11n（2.6M 参数）与扩展模型 YOLO11s（9.4M 参数）的精度-速度 Pareto 权衡。
4. 端到端决策闭环：
   检测框后级联市政养护分级引擎，直接输出 HIGH / MEDIUM / LOW 抢修派单建议，并搭载实时行车记录仪 HUD 视频推断演示（实测达 38.6 FPS，严格满足 30 FPS 车载实时要求）。
5. 压力测试：
   构建包含低光照、运动模糊、雨雾散射、高感 CMOS 噪声 5 类工况的鲁棒性压力测试。

【团队分工说明 / Division of Labor】
团队目前规划了 5 大核心工作任务（模型架构与对比训练、数据工程与质检、跨域泛化与鲁棒性测试、市政规则引擎与视频Demo、硬件测速与报告答辩）。关于组内每位成员具体负责的模块分工，目前小组正在结合实验推进情况进行内部细化探讨，将在模型实验推进阶段敲定后正式向助教报备。

我们已将完整开题报告（含详细系统架构设计与时间表规划）整理为 2 页正式 PDF 附件《CDS540_Project_Topic_Proposal.pdf》，请您查阅。如有任何修改建议，期待您的指导！

此致 / Best regards,
CDS540 李杨尚、吴松芫、郭毅锦、宋婧仪、黄浠沿 团队全体
```

---

## 📌 版本二：微信 / 课程群聊速发精简版（适合即时通讯）

```text
助教老师您好！我们是 CDS540 小组，在此提交 10月16日 期末大作业课题申报：

【选题方向】Candidate Topic 3: Object Detection and Image Segmentation（道路损坏检测与运维分级）
【项目题目】Automated Road Damage Detection & Operational Triage for Smart City Infrastructure Maintenance（基于计算机视觉的智慧城市道路病害自动化检测与运维分级系统）
【小组成员（共5人）】
- 组长：李杨尚 (5586669, yangshangli@LN.hk)
- 组员：吴松芫 (5550024), 郭毅锦 (5585392), 宋婧仪 (5554757), 黄浠沿 (5594367)
【核心亮点】
1. 数据集：IEEE CRDDC RDD2022（19,208张图像，捷克/印度/美国），严格无泄漏划分 + 851张纯净背景帧 + 4,805张美国数据留作跨国泛化测试。
2. 模型与指标：YOLO11n Baseline vs YOLO11s 对比，重点优化市政责任非对称高危指标“坑洼召回率（Pothole Recall）”。
3. 端到端闭环：下游接入市政养护分级引擎（HIGH/MEDIUM/LOW派单）及 120 帧实时行车记录仪 HUD 视频 Demo（38.6 FPS 满足实时巡检）。
4. 分工说明：涵盖数据工程、模型训练、跨域及5类恶劣天气测试、规则与Demo、报告与答辩5大模块，组员具体个人分配正在组内细化，后续正式同步。

详细立项书请见随附的 PDF 文件《CDS540_Project_Topic_Proposal.pdf》，辛苦助教查阅！
```

---

## 📌 版本三：纯英文版（English Version）

```text
Dear TA,

On behalf of our group, I am writing to submit our project topic proposal for CDS540 Computer Vision according to the course timeline (Deadline: Oct 16):

1. Project Track: Candidate Topic 3 — Object Detection and Image Segmentation (Direction 2: Road Damage Detection)
2. Project Title: Automated Road Damage Detection & Operational Triage for Smart City Infrastructure Maintenance
3. Group Members:
   - Group Leader: Yangshang Li (Student ID: 5586669, Email: yangshangli@LN.hk)
   - Team Member: Songyuan Wu (Student ID: 5550024, Email: songyuanwu@ln.hk)
   - Team Member: Yijin Guo (Student ID: 5585392, Email: yijinguo@ln.hk)
   - Team Member: Jingyi Song (Student ID: 5554757, Email: jingyisong3@ln.hk)
   - Team Member: Xiyan Huang (Student ID: 5594367, Email: xiyanhuang@ln.hk)
4. Brief Abstract & Scope:
   - Problem & Motivation: Using low-cost fleet dashcams to automate road pavement distress inspection, optimizing specifically for the asymmetric liability cost of severe potholes (D40 recall).
   - Dataset & Integrity: Multi-national RDD2022 dataset (19,208 images across Czech Republic, India, US). Deterministic leakage-free partition with 851 clean road background frames to penalize false alarms, and 100% held-out US data (4,805 images) for zero-shot cross-continent domain transfer.
   - Comparative Models: Lightweight YOLO11n (2.6M params) vs scaled YOLO11s (9.4M params) Pareto frontier.
   - End-to-End Decision System: Integrated operational priority triage engine (HIGH / MEDIUM / LOW repair dispatch) paired with real-time HUD dashcam inference video telemetry (operating deterministically at 38.6 FPS on GPU).
   - Stress Testing: Benchmarking across 5 adverse environmental perturbations (Night, Motion Blur, Fog, Noise).
5. Team Division of Labor:
   We have established 5 core functional modules (Modeling, Data Engineering, Domain & Robustness, Priority Engine & Demo, Hardware Latency & Academic Writing). Individual module assignments are currently being finalized internally and will be formally submitted during the experiment phase.

Please find our formal 2-page project proposal PDF attached (`CDS540_Project_Topic_Proposal.pdf`). We look forward to your valuable feedback and guidance.

Best regards,
Yangshang Li and Team
CDS540 Computer Vision
```
