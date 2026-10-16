"""
Generates the bilingual (Chinese default / English toggle) interactive system architecture HTML dashboard.
Outputs to:
  - docs/system_architecture.html
  - submission/system_architecture.html
"""
import os
import json
import zipfile

def build_html():
    html_content = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CDS540 道路损伤检测与市政运维派工 - 交互式端到端系统架构看板</title>
  <style>
    :root {
      --bg-primary: #0a0e17;
      --bg-secondary: #111827;
      --bg-card: rgba(17, 24, 39, 0.88);
      --bg-glass: rgba(26, 38, 57, 0.70);
      --border-color: rgba(59, 130, 246, 0.28);
      --border-glow: rgba(59, 130, 246, 0.6);
      --text-main: #f3f4f6;
      --text-muted: #9ca3af;
      --accent-blue: #3b82f6;
      --accent-cyan: #06b6d4;
      --accent-green: #10b981;
      --accent-purple: #8b5cf6;
      --accent-amber: #f59e0b;
      --accent-red: #ef4444;
      --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      user-select: none;
    }

    body {
      background-color: var(--bg-primary);
      color: var(--text-main);
      font-family: var(--font-family);
      overflow: hidden;
      width: 100vw;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }

    /* Top Navigation Header */
    header {
      height: 64px;
      background: rgba(10, 14, 23, 0.94);
      backdrop-filter: blur(14px);
      border-bottom: 1px solid var(--border-color);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 24px;
      z-index: 100;
      box-shadow: 0 4px 24px rgba(0, 0, 0, 0.5);
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
    }

    .brand-badge {
      background: linear-gradient(135deg, #1e3a8a, #3b82f6);
      color: white;
      font-size: 11px;
      font-weight: 700;
      padding: 5px 10px;
      border-radius: 6px;
      letter-spacing: 0.5px;
      border: 1px solid rgba(255, 255, 255, 0.25);
      box-shadow: 0 0 12px rgba(59, 130, 246, 0.35);
    }

    .brand-title {
      font-size: 16px;
      font-weight: 700;
      letter-spacing: -0.2px;
      background: linear-gradient(90deg, #ffffff, #93c5fd);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .brand-subtitle {
      font-size: 11px;
      color: var(--text-muted);
      margin-top: 2px;
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .btn {
      background: var(--bg-glass);
      color: var(--text-main);
      border: 1px solid var(--border-color);
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .btn:hover {
      background: rgba(59, 130, 246, 0.2);
      border-color: var(--accent-blue);
      box-shadow: 0 0 12px rgba(59, 130, 246, 0.3);
      transform: translateY(-1px);
    }

    .btn-lang {
      background: linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(59, 130, 246, 0.2));
      border-color: rgba(139, 92, 246, 0.4);
      color: #c4b5fd;
    }
    .btn-lang:hover {
      background: linear-gradient(135deg, rgba(139, 92, 246, 0.35), rgba(59, 130, 246, 0.35));
      border-color: #a78bfa;
      color: #ffffff;
    }

    .btn-primary {
      background: linear-gradient(135deg, #2563eb, #1d4ed8);
      border-color: #3b82f6;
      color: #ffffff;
      box-shadow: 0 0 12px rgba(37, 99, 235, 0.4);
    }
    .btn-primary:hover {
      background: linear-gradient(135deg, #3b82f6, #2563eb);
      box-shadow: 0 0 18px rgba(59, 130, 246, 0.6);
    }

    /* Interactive World Canvas */
    #viewport {
      flex: 1;
      position: relative;
      overflow: hidden;
      cursor: grab;
      background: radial-gradient(circle at 50% 50%, #111827 0%, #0a0e17 100%);
    }

    #viewport:active {
      cursor: grabbing;
    }

    .grid-bg {
      position: absolute;
      top: 0; left: 0; right: 0; bottom: 0;
      background-image: 
        linear-gradient(rgba(59, 130, 246, 0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(59, 130, 246, 0.05) 1px, transparent 1px);
      background-size: 40px 40px;
      pointer-events: none;
    }

    #world {
      position: absolute;
      top: 0;
      left: 0;
      transform-origin: 0 0;
      width: 2600px;
      height: 1600px;
    }

    /* SVG Connection Lines */
    #connections-layer {
      position: absolute;
      top: 0;
      left: 0;
      width: 2600px;
      height: 1600px;
      pointer-events: none;
      z-index: 10;
    }

    path.wire {
      fill: none;
      stroke: rgba(96, 165, 250, 0.35);
      stroke-width: 2.2;
      stroke-dasharray: 6, 6;
      animation: flowAnim 1.6s linear infinite;
      transition: stroke 0.3s, stroke-width 0.3s;
    }

    path.wire-active {
      stroke: #38bdf8;
      stroke-width: 3.5;
      filter: drop-shadow(0 0 8px rgba(56, 189, 248, 0.8));
    }

    @keyframes flowAnim {
      from { stroke-dashoffset: 24; }
      to { stroke-dashoffset: 0; }
    }

    /* HTML Nodes & Tiers */
    #nodes-layer {
      position: absolute;
      top: 0;
      left: 0;
      width: 2600px;
      height: 1600px;
      z-index: 20;
    }

    .tier-box {
      position: absolute;
      border: 1px dashed rgba(255, 255, 255, 0.14);
      background: rgba(17, 24, 39, 0.35);
      border-radius: 16px;
      pointer-events: none;
      box-shadow: inset 0 0 40px rgba(0, 0, 0, 0.3);
    }

    .tier-label {
      position: absolute;
      top: 14px;
      left: 18px;
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.8px;
      text-transform: uppercase;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    /* Node Cards */
    .node-card {
      position: absolute;
      width: 240px;
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 14px;
      cursor: pointer;
      backdrop-filter: blur(8px);
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
      transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
      pointer-events: auto;
    }

    .node-card:hover {
      transform: translateY(-4px) scale(1.02);
      border-color: var(--accent-blue);
      box-shadow: 0 10px 30px rgba(59, 130, 246, 0.35);
      z-index: 50;
    }

    .node-card.active {
      border-color: #38bdf8;
      box-shadow: 0 0 30px rgba(56, 189, 248, 0.6);
      z-index: 60;
    }

    .node-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 8px;
    }

    .node-type-tag {
      font-size: 9px;
      font-weight: 800;
      padding: 3px 6px;
      border-radius: 4px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .tag-data   { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }
    .tag-proc   { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .tag-model  { background: rgba(139, 92, 246, 0.2); color: #c084fc; border: 1px solid rgba(139, 92, 246, 0.4); }
    .tag-eval   { background: rgba(236, 72, 153, 0.2); color: #f472b6; border: 1px solid rgba(236, 72, 153, 0.4); }
    .tag-biz    { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .tag-deploy { background: rgba(6, 182, 212, 0.2); color: #22d3ee; border: 1px solid rgba(6, 182, 212, 0.4); }

    .node-title {
      font-size: 13px;
      font-weight: 700;
      color: #ffffff;
      margin-bottom: 4px;
      line-height: 1.3;
    }

    .node-desc {
      font-size: 11px;
      color: var(--text-muted);
      line-height: 1.4;
      margin-bottom: 8px;
      height: 32px;
      overflow: hidden;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
    }

    .node-metric {
      background: rgba(0, 0, 0, 0.35);
      border-radius: 6px;
      padding: 5px 8px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .metric-label {
      font-size: 10px;
      color: var(--text-muted);
    }

    .metric-val {
      font-size: 11px;
      font-weight: 700;
      color: var(--accent-cyan);
    }

    /* Floating Viewport Controls */
    .controls-hud {
      position: absolute;
      bottom: 24px;
      right: 24px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      z-index: 80;
    }

    .hud-btn {
      width: 38px;
      height: 38px;
      border-radius: 8px;
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      color: var(--text-main);
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      font-size: 16px;
      font-weight: bold;
      backdrop-filter: blur(10px);
      transition: all 0.2s;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }

    .hud-btn:hover {
      background: rgba(59, 130, 246, 0.25);
      border-color: var(--accent-blue);
      color: white;
      transform: scale(1.08);
    }

    /* Slide-out Component Inspector Drawer */
    #drawer {
      position: fixed;
      top: 64px;
      right: -460px;
      width: 440px;
      height: calc(100vh - 64px - 44px);
      background: rgba(13, 20, 36, 0.95);
      backdrop-filter: blur(16px);
      border-left: 1px solid var(--border-color);
      box-shadow: -8px 0 32px rgba(0, 0, 0, 0.6);
      transition: right 0.35s cubic-bezier(0.16, 1, 0.3, 1);
      z-index: 90;
      padding: 24px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }

    #drawer.open {
      right: 0;
    }

    .drawer-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .drawer-close {
      background: none;
      border: none;
      color: var(--text-muted);
      font-size: 22px;
      cursor: pointer;
      padding: 4px;
      line-height: 1;
    }

    .drawer-close:hover {
      color: white;
    }

    .drawer-title {
      font-size: 18px;
      font-weight: 800;
      color: #ffffff;
      line-height: 1.3;
    }

    .drawer-path {
      font-family: monospace;
      font-size: 11px;
      color: var(--accent-cyan);
      background: rgba(6, 182, 212, 0.1);
      padding: 4px 8px;
      border-radius: 4px;
      border: 1px solid rgba(6, 182, 212, 0.2);
      display: inline-block;
      word-break: break-all;
    }

    .drawer-section {
      background: rgba(17, 24, 39, 0.7);
      border-radius: 8px;
      padding: 14px;
      border: 1px solid rgba(255, 255, 255, 0.06);
    }

    .drawer-section-title {
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      color: var(--accent-blue);
      letter-spacing: 0.5px;
      margin-bottom: 8px;
    }

    .drawer-body {
      font-size: 12px;
      line-height: 1.6;
      color: #d1d5db;
    }

    .drawer-metric-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
    }

    .drawer-metric-card {
      background: rgba(0, 0, 0, 0.3);
      padding: 10px;
      border-radius: 6px;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .dm-val {
      font-size: 16px;
      font-weight: 800;
      color: var(--accent-cyan);
    }

    .dm-lbl {
      font-size: 10px;
      color: var(--text-muted);
      margin-top: 2px;
    }

    /* Bottom Global Telemetry Ribbon */
    footer {
      height: 44px;
      background: rgba(10, 14, 23, 0.94);
      backdrop-filter: blur(14px);
      border-top: 1px solid var(--border-color);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 24px;
      font-size: 11px;
      color: var(--text-muted);
      z-index: 100;
    }

    .stat-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      margin-right: 18px;
    }

    .stat-val {
      font-weight: 700;
      color: #ffffff;
    }
  </style>
</head>
<body>

  <!-- Top Header Navigation -->
  <header>
    <div class="brand">
      <span class="brand-badge" id="ui-brand-badge">CDS540 计算机视觉</span>
      <div>
        <div class="brand-title" id="ui-brand-title">RDD2022 智慧城市道路损伤检测流水线</div>
        <div class="brand-subtitle" id="ui-brand-subtitle">端到端系统架构、数据流向拓扑与端侧部署矩阵</div>
      </div>
    </div>
    <div class="header-actions">
      <button class="btn btn-lang" onclick="toggleLanguage()">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>
        <span id="ui-btn-lang">English / 切换英文</span>
      </button>
      <button class="btn" onclick="resetView()">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
        <span id="ui-btn-reset">复位全景</span>
      </button>
      <button class="btn" id="btn-anim" onclick="toggleAnimation()">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>
        <span id="ui-btn-anim">暂停动效</span>
      </button>
      <button class="btn btn-primary" onclick="window.open('../submission/CDS540_Final_Project_Report.pdf')">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
        <span id="ui-btn-report">查阅 6 页正规结题报告 (PDF)</span>
      </button>
    </div>
  </header>

  <!-- Interactive World Viewport -->
  <div id="viewport">
    <div class="grid-bg"></div>

    <div id="world">
      <!-- SVG Wire Connection Layer -->
      <svg id="connections-layer" viewBox="0 0 2600 1600"></svg>

      <!-- HTML Nodes & Tiers Layer -->
      <div id="nodes-layer"></div>
    </div>

    <!-- Floating Zoom/Pan Controls HUD -->
    <div class="controls-hud">
      <button class="hud-btn" onclick="zoomIn()" title="放大 (+)">+</button>
      <button class="hud-btn" onclick="zoomOut()" title="缩小 (−)">−</button>
      <button class="hud-btn" onclick="resetView()" title="适配居中 (⤢)">⤢</button>
    </div>
  </div>

  <!-- Slide-out Component Inspector Drawer -->
  <div id="drawer">
    <div class="drawer-header">
      <span class="node-type-tag tag-model" id="d-tag">COMPONENT</span>
      <button class="drawer-close" onclick="closeDrawer()">&times;</button>
    </div>
    <div class="drawer-title" id="d-title">组件详情</div>
    <div class="drawer-path" id="d-path">scripts/module.py</div>

    <div class="drawer-section">
      <div class="drawer-section-title" id="d-sec-role">核心职责与系统角色</div>
      <div class="drawer-body" id="d-desc">描述信息...</div>
    </div>

    <div class="drawer-section" id="d-sec-tech">
      <div class="drawer-section-title" id="d-sec-spec">技术规范与数学算法设计</div>
      <div class="drawer-body" id="d-tech">技术细节...</div>
    </div>

    <div class="drawer-section">
      <div class="drawer-section-title" id="d-sec-bench">定量基准实验指标</div>
      <div class="drawer-metric-grid" id="d-metrics">
        <!-- Injected dynamically -->
      </div>
    </div>

    <div class="drawer-section">
      <div class="drawer-section-title" id="d-sec-link">管线上下游链路拓扑</div>
      <div class="drawer-body" style="font-size: 11px;">
        <p><b id="d-lbl-in">输入来源:</b> <span id="d-inputs" style="color: #93c5fd;">无</span></p>
        <p style="margin-top: 4px;"><b id="d-lbl-out">输出去向:</b> <span id="d-outputs" style="color: #6ee7b7;">无</span></p>
      </div>
    </div>
  </div>

  <!-- Bottom Global Telemetry Ribbon -->
  <footer>
    <div id="footer-left">
      <span class="stat-pill"><span style="color: var(--accent-cyan);">●</span> <span id="f-stat-1-lbl">域内测试集:</span> <span class="stat-val">1,579 张图像</span></span>
      <span class="stat-pill"><span style="color: var(--accent-blue);">●</span> <span id="f-stat-2-lbl">跨国测试集 (美国):</span> <span class="stat-val">4,805 张图像</span></span>
      <span class="stat-pill"><span style="color: var(--accent-green);">●</span> <span id="f-stat-3-lbl">最优模型 mAP@50:</span> <span class="stat-val">0.2846 (+31.2%)</span></span>
      <span class="stat-pill"><span style="color: var(--accent-amber);">●</span> <span id="f-stat-4-lbl">严重坑洞 (D40) 召回率:</span> <span class="stat-val">39.74% (+19.2%)</span></span>
    </div>
    <div id="footer-right">
      <span class="stat-pill"><span id="f-stat-5-lbl">RTX 5070 测速:</span> <span class="stat-val" style="color: #a78bfa;">25.89 ms (38.6 FPS)</span></span>
      <span class="stat-pill"><span id="f-stat-6-lbl">显存占用:</span> <span class="stat-val" style="color: #f472b6;">137.9 MB (&lt;2%)</span></span>
      <span class="stat-pill"><span id="f-stat-7-lbl">推荐端侧硬件:</span> <span class="stat-val" style="color: #34d399;">NVIDIA Jetson Orin Nano (8GB)</span></span>
    </div>
  </footer>

  <script>
    let currentLang = 'zh'; // Default to Chinese

    const I18N_UI = {
      zh: {
        badge: "CDS540 计算机视觉",
        title: "RDD2022 智慧城市道路损伤检测流水线",
        subtitle: "端到端系统架构、数据流向拓扑与端侧部署矩阵",
        btnLang: "English / 切换英文",
        btnReset: "复位全景",
        btnAnimPause: "暂停动效",
        btnAnimResume: "恢复动效",
        btnReport: "查阅 6 页正规结题报告 (PDF)",
        secRole: "核心职责与系统角色",
        secSpec: "技术规范与数学算法设计",
        secBench: "定量基准实验指标",
        secLink: "管线上下游链路拓扑",
        lblIn: "输入来源:",
        lblOut: "输出去向:",
        stat1Lbl: "域内测试集:",
        stat2Lbl: "跨国测试集 (美国):",
        stat3Lbl: "最优模型 mAP@50:",
        stat4Lbl: "严重坑洞 (D40) 召回率:",
        stat5Lbl: "RTX 5070 测速:",
        stat6Lbl: "显存占用:",
        stat7Lbl: "推荐端侧硬件:",
        tagMap: {
          data: "数据语料",
          proc: "处理算法",
          model: "检测模型",
          eval: "评测基准",
          biz: "市政决策",
          deploy: "硬件部署"
        }
      },
      en: {
        badge: "CDS540 COMPUTER VISION",
        title: "RDD2022 Smart City Road Damage Detection Pipeline",
        subtitle: "End-to-End System Architecture, Dataflow & Edge Deployment Matrix",
        btnLang: "中文 / Switch to Chinese",
        btnReset: "Reset View",
        btnAnimPause: "Pause Flow",
        btnAnimResume: "Resume Flow",
        btnReport: "Open 6-Page Report (PDF)",
        secRole: "Core Responsibility & Role",
        secSpec: "Technical Specifications & Formulation",
        secBench: "Quantitative Benchmarks",
        secLink: "Pipeline Linkage",
        lblIn: "Inputs:",
        lblOut: "Outputs:",
        stat1Lbl: "In-Domain Test:",
        stat2Lbl: "Held-Out Domain (US):",
        stat3Lbl: "Best Model mAP@50:",
        stat4Lbl: "Critical Pothole Recall:",
        stat5Lbl: "RTX 5070 E2E Latency:",
        stat6Lbl: "VRAM Allocated:",
        stat7Lbl: "Edge Target Platform:",
        tagMap: {
          data: "DATA",
          proc: "PROC",
          model: "MODEL",
          eval: "EVAL",
          biz: "DECISION",
          deploy: "DEPLOY"
        }
      }
    };

    // System Architecture Tiers Definition
    const TIERS = [
      { id: "T1", label_zh: "第 1 层：数据工程与防泄漏隔离", label_en: "Tier 1: Data Engineering & Leakage Isolation", x: 40, y: 50, w: 780, h: 620, color: "#3b82f6" },
      { id: "T2", label_zh: "第 2 层：模型架构设计与缩放机制", label_en: "Tier 2: Model Architecture & Scaling Regimes", x: 860, y: 50, w: 560, h: 620, color: "#8b5cf6" },
      { id: "T3", label_zh: "第 3 层：域内基准 vs 跨国泛化与压力测试", label_en: "Tier 3: In-Domain vs. Cross-Country Stress Testing", x: 1460, y: 50, w: 820, h: 620, color: "#ec4899" },
      { id: "T4", label_zh: "第 4 层：市政严重度启发式派工决策引擎", label_en: "Tier 4: Municipal Priority Triage Heuristic Engine", x: 40, y: 720, w: 780, h: 620, color: "#f59e0b" },
      { id: "T5", label_zh: "第 5 层：端侧实时视频推断与 HUD 仪表盘", label_en: "Tier 5: Real-Time Edge Video Inference & Telemetry HUD", x: 860, y: 720, w: 680, h: 620, color: "#10b981" },
      { id: "T6", label_zh: "第 6 层：硬件性能画像与最终学术交付物", label_en: "Tier 6: Hardware Profiling & Academic Deliverables", x: 1580, y: 720, w: 700, h: 620, color: "#06b6d4" }
    ];

    const NODES = [
      // Tier 1: Data Engineering
      {
        id: "raw_data", tier: "T1", x: 70, y: 120, tag: "data",
        title_zh: "原始多国 RDD2022 数据集",
        title_en: "Raw Multi-Country RDD2022",
        path: "DGS2022数据/",
        desc_zh: "涵盖捷克、印度与美国的 19,208 张行车记录仪真实道路路面图像。",
        desc_en: "19,208 raw dashcam images across Czech Republic, India, and United States.",
        metricLabel_zh: "图像语料库总量", metricVal_zh: "19,208 帧",
        metricLabel_en: "Total Image Corpus", metricVal_en: "19,208 Frames",
        tech_zh: "跨国多源道路影像，覆盖多样沥青、水泥混凝土板接缝及多季节阴雨复杂光照环境。",
        tech_en: "Multi-country road surface imagery covering diverse asphalt, concrete, and seasonal lighting conditions.",
        metrics_zh: [ { lbl: "原始图像总数", val: "19,208 张" }, { lbl: "标注文件", val: "15,340 份 XML" } ],
        metrics_en: [ { lbl: "Raw Images", val: "19,208" }, { lbl: "Annotations", val: "15,340 XMLs" } ],
        inputs: "None (Source Dataset)", outputs: "qa_inspector"
      },
      {
        id: "qa_inspector", tier: "T1", x: 330, y: 120, tag: "proc",
        title_zh: "标注坐标质量自检器",
        title_en: "Annotation QA Inspector",
        path: "scripts/annotation_qa_inspector.py",
        desc_zh: "校验所有目标检测框边界坐标，彻底排查坐标越界与逻辑颠倒样本。",
        desc_en: "Validates bounding box coordinates and eliminates labeling anomalies.",
        metricLabel_zh: "坐标异常检出数", metricVal_zh: "0 异常 (100% 合规)",
        metricLabel_en: "Coordinate Anomalies", metricVal_en: "0 Detected (100% Valid)",
        tech_zh: "遍历解析全部 20,962 个目标框，校验 x_min < x_max 且 y_min < y_max，保证坐标严格位于 [0, W] × [0, H] 物理边界内。",
        tech_en: "Parses all 20,962 bounding boxes. Audits x_min < x_max and y_min < y_max within legal image dimensions [0, W] x [0, H].",
        metrics_zh: [ { lbl: "解析边界框", val: "20,962 个" }, { lbl: "坐标越界错误", val: "0 个 (完美)" } ],
        metrics_en: [ { lbl: "Total Boxes", val: "20,962" }, { lbl: "Coordinate Errors", val: "0" } ],
        inputs: "raw_data", outputs: "splitter"
      },
      {
        id: "splitter", tier: "T1", x: 550, y: 120, tag: "proc",
        title_zh: "分层无数据泄漏切分器",
        title_en: "Stratified Dataset Splitter",
        path: "scripts/split_dataset.py",
        desc_zh: "采用固定随机种子 (seed = 42) 进行严格的图像级别隔离切分。",
        desc_en: "Strict image-level partitioning with fixed random seed (seed = 42).",
        metricLabel_zh: "数据隔离安全性", metricVal_zh: "0 数据泄漏 (Zero Leakage)",
        metricLabel_en: "Partition Isolation", metricVal_en: "Zero Leakage",
        tech_zh: "对捷克与印度语料进行 70/15/15 分层切分：训练集 7,375 帧、验证集 1,581 帧、域内测试集 1,579 帧。将美国数据 100% 独立保留作为跨国域测试。",
        tech_en: "Splits Czech & Indian annotated corpus: 7,375 Train (70%), 1,581 Val (15%), 1,579 Test (15%). Completely holds out US dataset.",
        metrics_zh: [ { lbl: "训练集图像", val: "7,375 张" }, { lbl: "验证集图像", val: "1,581 张" }, { lbl: "域内测试集", val: "1,579 张" } ],
        metrics_en: [ { lbl: "Train Images", val: "7,375" }, { lbl: "Val Images", val: "1,581" }, { lbl: "In-Domain Test", val: "1,579" } ],
        inputs: "qa_inspector", outputs: "yolo_converter, held_out_us"
      },
      {
        id: "held_out_us", tier: "T1", x: 330, y: 380, tag: "data",
        title_zh: "100% 独立保留的美国目标域",
        title_en: "100% Held-Out US Target Domain",
        path: "configs/data_domain_test.yaml",
        desc_zh: "4,805 张完全未参与训练的美国高速公路行车记录仪图像，严格物理隔离。",
        desc_en: "4,805 unseen United States highway dashcam images strictly isolated from training.",
        metricLabel_zh: "零样本跨国域测试", metricVal_zh: "4,805 张图像",
        metricLabel_en: "Zero-Shot Domain Test", metricVal_en: "4,805 Images",
        tech_zh: "在步骤 10 中专用于测试跨大洲公路材质（水泥混凝土板接缝）下的零样本泛化衰减。",
        tech_en: "Used exclusively in Step 10 for continental zero-shot transfer testing.",
        metrics_zh: [ { lbl: "目标域图像", val: "4,805 张" }, { lbl: "训练集污染率", val: "0% (纯零样本)" } ],
        metrics_en: [ { lbl: "Target Images", val: "4,805" }, { lbl: "Contamination", val: "0% (Pure Zero-Shot)" } ],
        inputs: "splitter", outputs: "domain_eval"
      },
      {
        id: "yolo_converter", tier: "T1", x: 550, y: 380, tag: "proc",
        title_zh: "NTFS 硬链接 VOC 转 YOLO 格式",
        title_en: "NTFS Hardlink VOC-to-YOLO",
        path: "scripts/voc_to_yolo.py",
        desc_zh: "将 VOC 绝对坐标转换为归一化 YOLO TXT，并完整保留 851 张无病害纯背景帧。",
        desc_en: "Converts VOC coordinates to normalized YOLO TXT with 851 negative background frames.",
        metricLabel_zh: "额外磁盘存储开销", metricVal_zh: "0 GB (NTFS 硬链接)",
        metricLabel_en: "Storage Overhead", metricVal_en: "0 GB (Hardlinks)",
        tech_zh: "保留 851 张完好路面负样本（0 字节空标注），强力惩罚道路标线、斑马线及树影上的假阳性虚警。",
        tech_en: "Preserves 851 background frames (clean sound roads) with 0-byte labels to penalize false alarms over asphalt markings.",
        metrics_zh: [ { lbl: "纯背景负样本", val: "851 帧 (53.9%)" }, { lbl: "最大坐标转换误差", val: "0.00048 px" } ],
        metrics_en: [ { lbl: "Negative Frames", val: "851 (53.9%)" }, { lbl: "Max Coord Error", val: "0.00048 px" } ],
        inputs: "splitter", outputs: "yolo11n, yolo11s"
      },

      // Tier 2: Model Architecture
      {
        id: "yolo11n", tier: "T2", x: 890, y: 150, tag: "model",
        title_zh: "YOLO11n 基线轻量架构",
        title_en: "YOLO11n Baseline Architecture",
        path: "models/baseline_best.pt",
        desc_zh: "紧凑型单阶段目标检测网络，针对边缘超低功耗微控制器优化。",
        desc_en: "Compact single-stage detector optimized for edge micro-controllers.",
        metricLabel_zh: "模型计算规模", metricVal_zh: "2.6M 参数 | 6.5 GFLOPs",
        metricLabel_en: "Model Footprint", metricVal_en: "2.6M Params | 6.5 GFLOPs",
        tech_zh: "在 RTX 5070 GPU 上使用 AdamW 优化器、余弦退火学习率 (lr=0.005) 及 FP16 混合精度训练 22 轮收敛。",
        tech_en: "Trained 22 epochs with AdamW, cosine annealing lr=0.005, mixed precision FP16 on RTX 5070 GPU.",
        metrics_zh: [ { lbl: "网络参数量", val: "2.6M" }, { lbl: "权重文件大小", val: "20.4 MB" }, { lbl: "单图浮点计算量", val: "6.5 GFLOPs" } ],
        metrics_en: [ { lbl: "Parameters", val: "2.6M" }, { lbl: "Model Size", val: "20.4 MB" }, { lbl: "GFLOPs", val: "6.5" } ],
        inputs: "yolo_converter", outputs: "in_domain_bench, domain_eval"
      },
      {
        id: "yolo11s", tier: "T2", x: 1140, y: 150, tag: "model",
        title_zh: "YOLO11s 缩放帕累托最优模型",
        title_en: "YOLO11s Scaled Pareto Model",
        path: "models/compare_yolo11s_best.pt",
        desc_zh: "加宽 C3k2 跨阶段瓶颈层，扩大感受野以捕捉微小裂缝与严重坑洞。",
        desc_en: "Wider C3k2 cross-stage bottlenecks with enhanced spatial receptive field.",
        metricLabel_zh: "高精度缩放版本", metricVal_zh: "9.4M 参数 | 21.5 GFLOPs",
        metricLabel_en: "High-Accuracy Scale", metricVal_en: "9.4M Params | 21.5 GFLOPs",
        tech_zh: "实现 mAP@50 跃升 +31.2%，严重坑洞 (D40) 召回率提升 +19.2%，端到端仅增加 3.9ms 延迟。",
        tech_en: "Delivers +31.2% higher mAP@50 and +19.2% higher pothole recall with only 3.9ms added latency.",
        metrics_zh: [ { lbl: "网络参数量", val: "9.4M" }, { lbl: "权重文件大小", val: "18.3 MB" }, { lbl: "单图浮点计算量", val: "21.5 GFLOPs" } ],
        metrics_en: [ { lbl: "Parameters", val: "9.4M" }, { lbl: "Model Size", val: "18.3 MB" }, { lbl: "GFLOPs", val: "21.5" } ],
        inputs: "yolo_converter", outputs: "in_domain_bench, domain_eval, rob_eval, priority_engine, runtime_profiler"
      },

      // Tier 3: Evaluation & Stress Testing
      {
        id: "in_domain_bench", tier: "T3", x: 1490, y: 140, tag: "eval",
        title_zh: "域内独立测试基准评估",
        title_en: "In-Domain Test Benchmark",
        path: "results/model_comparison.csv",
        desc_zh: "在 1,579 张独立的捷克与印度测试集图像上进行综合评测对比。",
        desc_en: "Evaluates on 1,579 independent test images (Czech Republic & India).",
        metricLabel_zh: "YOLO11s mAP@50", metricVal_zh: "0.2846 (+31.2% 提升)",
        metricLabel_en: "YOLO11s mAP@50", metricVal_en: "0.2846 (+31.2% Gain)",
        tech_zh: "YOLO11s 将关键危害坑洞 (D40) 的 AP 从 0.2093 提升至 0.2923，召回率从 33.3% 提升至 39.7%。",
        tech_en: "YOLO11s elevates D40 pothole AP from 0.2093 to 0.2923 and Pothole Recall from 33.3% to 39.7%.",
        metrics_zh: [ { lbl: "基线 YOLO11n mAP", val: "0.2169" }, { lbl: "缩放 YOLO11s mAP", val: "0.2846" }, { lbl: "D40 坑洞召回率", val: "39.74%" } ],
        metrics_en: [ { lbl: "YOLO11n mAP", val: "0.2169" }, { lbl: "YOLO11s mAP", val: "0.2846" }, { lbl: "D40 Recall", val: "39.74%" } ],
        inputs: "yolo11n, yolo11s", outputs: "report_gen"
      },
      {
        id: "domain_eval", tier: "T3", x: 1740, y: 140, tag: "eval",
        title_zh: "跨国域迁移泛化评估",
        title_en: "Cross-Country Domain Transfer",
        path: "scripts/evaluate_domain_shift.py",
        desc_zh: "在 4,805 张未参与训练的美国高速公路图像上进行零样本迁移测试。",
        desc_en: "Zero-shot evaluation on 4,805 unseen US highway images.",
        metricLabel_zh: "纵向裂缝衰减", metricVal_zh: "-4.98% (极高几何不变性)",
        metricLabel_en: "Longitudinal Decay", metricVal_en: "-4.98% (High Invariance)",
        tech_zh: "单轨纵向裂缝 (D00) 仅衰减 -4.98%；而网状龟裂 (D20) 暴跌 -82%，因美国混凝土路面板接缝与沥青灌缝胶产生强干扰。",
        tech_en: "Single-track longitudinal cracks (D00) decay by only -4.98%. Alligator cracking (D20) drops by -82% due to US concrete joints.",
        metrics_zh: [ { lbl: "美国测试图像", val: "4,805 张" }, { lbl: "总体 mAP 衰减", val: "-69.1%" }, { lbl: "D00 纵向裂缝衰减", val: "-4.98%" } ],
        metrics_en: [ { lbl: "US Test Images", val: "4,805" }, { lbl: "Overall Decay", val: "-69.1%" }, { lbl: "D00 Decay", val: "-4.98%" } ],
        inputs: "held_out_us, yolo11s", outputs: "report_gen"
      },
      {
        id: "rob_eval", tier: "T3", x: 1990, y: 140, tag: "eval",
        title_zh: "恶劣运营环境鲁棒性压力测试",
        title_en: "Environmental Stress Testing",
        path: "scripts/test_robustness.py",
        desc_zh: "模拟 5 种实际工况扰动：低照度夜间、行车运动模糊、浓雾天气、传感器高斯噪声。",
        desc_en: "Simulates adverse operating conditions: Low-Light, Blur, Fog, Noise.",
        metricLabel_zh: "运动模糊不对称衰减", metricVal_zh: "裂缝 -87% | 坑洞仅 -18%",
        metricLabel_en: "Motion Blur Decay", metricVal_en: "Cracks -87% | Potholes -18%",
        tech_zh: "水平运动模糊完全破坏了 1D 线性裂缝的高频边缘梯度；但 2D 凹陷坑洞因具备大面积凹陷阴影，依然保留充足的空间对比度。",
        tech_en: "Directional motion blur wipes out linear 1D crack edges, but 2D pothole depressions retain sufficient spatial contrast.",
        metrics_zh: [ { lbl: "清晰基准 mAP", val: "0.2978" }, { lbl: "夜间暗光 mAP", val: "0.2060" }, { lbl: "模糊下坑洞召回率", val: "35.58%" } ],
        metrics_en: [ { lbl: "Clean mAP", val: "0.2978" }, { lbl: "Night mAP", val: "0.2060" }, { lbl: "Blur Pothole Recall", val: "35.58%" } ],
        inputs: "yolo11s", outputs: "report_gen"
      },

      // Tier 4: Priority Decision Engine
      {
        id: "priority_engine", tier: "T4", x: 80, y: 820, tag: "biz",
        title_zh: "市政严重度启发式派工决策引擎",
        title_en: "Municipal Priority Triage Engine",
        path: "scripts/priority.py",
        desc_zh: "将检测框几何坐标与类别置信度转化为市政运维优先级派工评分。",
        desc_en: "Translates raw bounding box detections into municipal repair dispatch priorities.",
        metricLabel_zh: "危害性惩罚权重", metricVal_zh: "严重坑洞 D40 = 10.0×",
        metricLabel_en: "Hazard Weighting", metricVal_en: "Pothole D40 = 10.0x",
        tech_zh: "综合严重度 = 缺陷多样性 × ∑(危害权重 × 置信度 × 面积归一化因子)。危害权重：D40=10.0, D20=6.5, D10=3.5, D00=2.5。",
        tech_en: "Severity = Multiplicity * Sum(Hazard_Weight * Confidence * Area_Penalty). D40=10.0, D20=6.5, D10=3.5, D00=2.5.",
        metrics_zh: [ { lbl: "D40 坑洞权重", val: "10.0 (极高法律与爆胎风险)" }, { lbl: "D20 龟裂权重", val: "6.5" }, { lbl: "单帧运算开销", val: "2.24 ms" } ],
        metrics_en: [ { lbl: "D40 Weight", val: "10.0 (High Risk)" }, { lbl: "D20 Weight", val: "6.5" }, { lbl: "Compute Latency", val: "2.24 ms" } ],
        inputs: "yolo11s", outputs: "triage_tiers, edge_demo"
      },
      {
        id: "triage_tiers", tier: "T4", x: 420, y: 820, tag: "biz",
        title_zh: "市政四级闭环抢修响应梯队",
        title_en: "Operational Maintenance Tiers",
        path: "reports/priority_rules.md",
        desc_zh: "将评分映射到具体的市政道路养护施工抢修工单标准。",
        desc_en: "Categorizes into actionable municipal repair orders.",
        metricLabel_zh: "抢修调度行动", metricVal_zh: "高危 (HIGH): 24h 内冷补沥青",
        metricLabel_en: "Dispatch Action", metricVal_en: "HIGH: 24h Cold-Patch",
        tech_zh: "高危 HIGH (评分 ≥ 12.0，24-48h 内应急冷补)、中危 MEDIUM (评分 ≥ 6.0，30天内热拌沥青铣刨重铺)、低危 LOW (评分 ≥ 0.1，预防性灌缝养护)、无病害 NONE (良好路面)。",
        tech_en: "HIGH (Emergency patch in 24-48h), MEDIUM (30-day mill/pave queue), LOW (Preventive crack seal), NONE (Sound pavement).",
        metrics_zh: [ { lbl: "高危等级阈值", val: "评分 ≥ 12.0" }, { lbl: "中危等级阈值", val: "评分 ≥ 6.0" }, { lbl: "低危等级阈值", val: "评分 ≥ 0.1" } ],
        metrics_en: [ { lbl: "HIGH Tier", val: "Score >= 12.0" }, { lbl: "MEDIUM Tier", val: "Score >= 6.0" }, { lbl: "LOW Tier", val: "Score >= 0.1" } ],
        inputs: "priority_engine", outputs: "edge_demo"
      },

      // Tier 5: Edge Video Demo & HUD
      {
        id: "edge_demo", tier: "T5", x: 890, y: 820, tag: "deploy",
        title_zh: "端侧实时视频推断与 HUD 仪表盘",
        title_en: "Real-Time Video Demo Engine",
        path: "demo/demo.py",
        desc_zh: "逐帧摄入车载视频，执行损伤检测并动态叠加半透明平视显示 HUD 遥测横幅。",
        desc_en: "Ingests video frames, runs detection, and overlays live HUD telemetry banner.",
        metricLabel_zh: "实战演示视频输出", metricVal_zh: "120 帧 @ 15 FPS",
        metricLabel_en: "Demo Video Output", metricVal_en: "120 Frames @ 15 FPS",
        tech_zh: "实时绘制高亮边界框并在画面顶部渲染 HUD 看板：展示当前应急响应级别、各类型病害实时计数及端到端运算延迟。",
        tech_en: "Draws bounding boxes and dynamic top HUD dashboard: Priority Alert Badge, Defect Counts, and Live Latency.",
        metrics_zh: [ { lbl: "输入行车记录流", val: "1280×720 分辨率" }, { lbl: "HUD 遥测看板", val: "实时动态渲染" } ],
        metrics_en: [ { lbl: "Input Stream", val: "1280x720 Dashcam" }, { lbl: "HUD Telemetry", val: "Real-Time Active" } ],
        inputs: "priority_engine, triage_tiers", outputs: "demo_video"
      },
      {
        id: "demo_video", tier: "T5", x: 1210, y: 820, tag: "deploy",
        title_zh: "道路巡检实战视频 Demo 片段",
        title_en: "Inspection Video Demo Clip",
        path: "demo/output.mp4",
        desc_zh: "导出的 8.0 秒 MP4 高清演示视频，可直接内嵌于 Moodle 答辩幻灯片中播放。",
        desc_en: "Exported 8.0-second demonstration video ready for Moodle presentation embedding.",
        metricLabel_zh: "视频文件体积", metricVal_zh: "11.2 MB MP4",
        metricLabel_en: "File Size", metricVal_en: "11.2 MB MP4",
        tech_zh: "全高清可视化呈现系统从远景检测、出现坑洞时 HUD 从 LOW 瞬间警报切换至 HIGH 的全流程。",
        tech_en: "Full HD visualization demonstrating real-time alert switching from LOW to HIGH priority as potholes appear.",
        metrics_zh: [ { lbl: "视频时长", val: "8.0 秒" }, { lbl: "视频编码格式", val: "720p H.264" } ],
        metrics_en: [ { lbl: "Duration", val: "8.0 seconds" }, { lbl: "Resolution", val: "720p H.264" } ],
        inputs: "edge_demo", outputs: "presentation_deck"
      },

      // Tier 6: Hardware Profiling & Deliverables
      {
        id: "runtime_profiler", tier: "T6", x: 1610, y: 820, tag: "deploy",
        title_zh: "高精度 CUDA Event 硬件测速画像",
        title_en: "Precision CUDA Event Profiler",
        path: "scripts/benchmark_runtime.py",
        desc_zh: "在 NVIDIA GeForce RTX 5070 笔记本 GPU 上进行严格纳秒级硬件延迟测速。",
        desc_en: "Rigorous hardware latency profiling on NVIDIA RTX 5070 Laptop GPU.",
        metricLabel_zh: "端到端 P95 延迟", metricVal_zh: "29.61 ms (< 33.3ms 预算)",
        metricLabel_en: "E2E P95 Latency", metricVal_en: "29.61 ms (< 33.3ms Budget)",
        tech_zh: "30 次热身后执行 100 次严格 CUDA 同步测速。全流水线稳定达到 38.6 FPS，峰值显存仅占用 137.9 MB。",
        tech_en: "100 iterations with 30 warmups. Full pipeline achieves 38.6 FPS with only 137.9 MB peak VRAM.",
        metrics_zh: [ { lbl: "纯网络前向耗时", val: "23.5 ms" }, { lbl: "端到端 P95 延迟", val: "29.61 ms" }, { lbl: "实时处理吞吐量", val: "38.6 FPS" } ],
        metrics_en: [ { lbl: "Pure Forward", val: "23.5 ms" }, { lbl: "E2E P95", val: "29.61 ms" }, { lbl: "Throughput", val: "38.6 FPS" } ],
        inputs: "yolo11s", outputs: "hardware_target"
      },
      {
        id: "hardware_target", tier: "T6", x: 1950, y: 820, tag: "deploy",
        title_zh: "推荐部署平台：Jetson Orin Nano",
        title_en: "Target Platform: Jetson Orin Nano",
        path: "reports/runtime_deployment_report.md",
        desc_zh: "针对市政巡逻车与公交车队推荐的工业级边缘计算部署硬件选型。",
        desc_en: "Recommended edge deployment hardware for municipal fleet inspection vehicles.",
        metricLabel_zh: "端侧算力规格", metricVal_zh: "8GB 统一内存 | 7-15W 极低功耗",
        metricLabel_en: "Edge Compute Spec", metricVal_en: "8GB VRAM | 7-15W TDP",
        tech_zh: "整机成本低于 $500，彻底消除每月每车超 $100 的 4G/5G 原始视频回传流量费，在偏远无信号山区 100% 离线自主运行。",
        tech_en: "Under $500, eliminates 4G/5G streaming fees ($100+/mo/vehicle), operates 100% offline in rural dead zones.",
        metrics_zh: [ { lbl: "端侧峰值算力", val: "40-67 TOPS" }, { lbl: "最大整机功耗", val: "15W Max" } ],
        metrics_en: [ { lbl: "Compute Power", val: "40-67 TOPS" }, { lbl: "Power Draw", val: "15W Max" } ],
        inputs: "runtime_profiler", outputs: "academic_report, presentation_deck"
      },
      {
        id: "academic_report", tier: "T6", x: 1610, y: 1080, tag: "proc",
        title_zh: "官方 6 页正规学术报告 (PDF)",
        title_en: "Official 6-Page Academic PDF",
        path: "submission/CDS540_Final_Project_Report.pdf",
        desc_zh: "严格满足 Times New Roman 12pt、1.25 倍行距及 ≤ 8 页严格页数限制的结题论文。",
        desc_en: "Formal project report strictly complying with Times New Roman 12pt and <= 8 pages limit.",
        metricLabel_zh: "论文精确页数", metricVal_zh: "恰好 6 页 (严格 ≤ 8 页)",
        metricLabel_en: "Page Count", metricVal_en: "Exactly 6 Pages (<=8)",
        tech_zh: "基于 ReportLab 编译生成。包含 4 个正式学术三线表、5 张 300 DPI 嵌入式图表及动态页眉页脚。",
        tech_en: "Compiled via ReportLab. Contains 4 formal academic tables, 5 embedded 300 DPI figures, and dynamic running headers.",
        metrics_zh: [ { lbl: "正文字体", val: "Times-Roman 12pt" }, { lbl: "标准行距", val: "1.25x (15pt)" }, { lbl: "学术表格", val: "4 个正式三线表" } ],
        metrics_en: [ { lbl: "Font", val: "Times-Roman 12pt" }, { lbl: "Line Spacing", val: "1.25x (15pt)" }, { lbl: "Tables", val: "4 Tables" } ],
        inputs: "in_domain_bench, domain_eval, rob_eval, hardware_target", outputs: "None (Final Deliverable)"
      },
      {
        id: "presentation_deck", tier: "T6", x: 1950, y: 1080, tag: "proc",
        title_zh: "18 页原生 16:9 汇报幻灯片 (PPTX)",
        title_en: "18-Slide Native PowerPoint Deck",
        path: "submission/CDS540_Final_Presentation_Deck.pptx",
        desc_zh: "符合 17 分钟汇报架构的终版演示文稿，内嵌高清图表与完整逐字演讲稿。",
        desc_en: "Official 17-minute presentation deck with embedded figures and verbatim speaker scripts.",
        metricLabel_zh: "幻灯片总页数", metricVal_zh: "18 页 (16:9 宽屏)",
        metricLabel_en: "Slide Count", metricVal_en: "18 Slides (16:9 Widescreen)",
        tech_zh: "基于 python-pptx 编译生成。在演讲者视图中为 4 位汇报组员注入了逐字逐句的英文演讲台词。",
        tech_en: "Generated via python-pptx. Full speaker notes populated in Presenter View for 4-speaker team delivery.",
        metrics_zh: [ { lbl: "演说汇报控制", val: "17 分钟汇报 + 3 分钟答辩" }, { lbl: "高清嵌入图表", val: "5 张高分辨率图" } ],
        metrics_en: [ { lbl: "Timing Budget", val: "17 min + 3 min Q&A" }, { lbl: "High-Res Figures", val: "5 Figures" } ],
        inputs: "demo_video, hardware_target", outputs: "None (Final Deliverable)"
      }
    ];

    const CONNECTIONS = [
      { from: "raw_data", to: "qa_inspector" },
      { from: "qa_inspector", to: "splitter" },
      { from: "splitter", to: "held_out_us" },
      { from: "splitter", to: "yolo_converter" },
      { from: "yolo_converter", to: "yolo11n" },
      { from: "yolo_converter", to: "yolo11s" },
      { from: "yolo11n", to: "in_domain_bench" },
      { from: "yolo11s", to: "in_domain_bench" },
      { from: "held_out_us", to: "domain_eval" },
      { from: "yolo11s", to: "domain_eval" },
      { from: "yolo11s", to: "rob_eval" },
      { from: "yolo11s", to: "priority_engine" },
      { from: "priority_engine", to: "triage_tiers" },
      { from: "priority_engine", to: "edge_demo" },
      { from: "triage_tiers", to: "edge_demo" },
      { from: "edge_demo", to: "demo_video" },
      { from: "yolo11s", to: "runtime_profiler" },
      { from: "runtime_profiler", to: "hardware_target" },
      { from: "in_domain_bench", to: "academic_report" },
      { from: "hardware_target", to: "academic_report" },
      { from: "demo_video", to: "presentation_deck" },
      { from: "hardware_target", to: "presentation_deck" }
    ];

    // Viewport Pan & Zoom State
    let scale = 0.52;
    let panX = 30;
    let panY = 15;
    let isDragging = false;
    let startX = 0;
    let startY = 0;
    let activeWireAnim = true;
    let selectedNodeObj = null;

    const viewport = document.getElementById('viewport');
    const world = document.getElementById('world');
    const nodesLayer = document.getElementById('nodes-layer');
    const connLayer = document.getElementById('connections-layer');

    // Language Toggle
    function toggleLanguage() {
      currentLang = currentLang === 'zh' ? 'en' : 'zh';
      applyLanguage();
    }

    function applyLanguage() {
      const t = I18N_UI[currentLang];
      document.getElementById('ui-brand-badge').innerText = t.badge;
      document.getElementById('ui-brand-title').innerText = t.title;
      document.getElementById('ui-brand-subtitle').innerText = t.subtitle;
      document.getElementById('ui-btn-lang').innerText = t.btnLang;
      document.getElementById('ui-btn-reset').innerText = t.btnReset;
      document.getElementById('ui-btn-report').innerText = t.btnReport;
      document.getElementById('ui-btn-anim').innerText = activeWireAnim ? t.btnAnimPause : t.btnAnimResume;

      document.getElementById('d-sec-role').innerText = t.secRole;
      document.getElementById('d-sec-spec').innerText = t.secSpec;
      document.getElementById('d-sec-bench').innerText = t.secBench;
      document.getElementById('d-sec-link').innerText = t.secLink;
      document.getElementById('d-lbl-in').innerText = t.lblIn;
      document.getElementById('d-lbl-out').innerText = t.lblOut;

      document.getElementById('f-stat-1-lbl').innerText = t.stat1Lbl;
      document.getElementById('f-stat-2-lbl').innerText = t.stat2Lbl;
      document.getElementById('f-stat-3-lbl').innerText = t.stat3Lbl;
      document.getElementById('f-stat-4-lbl').innerText = t.stat4Lbl;
      document.getElementById('f-stat-5-lbl').innerText = t.stat5Lbl;
      document.getElementById('f-stat-6-lbl').innerText = t.stat6Lbl;
      document.getElementById('f-stat-7-lbl').innerText = t.stat7Lbl;

      // Update Tiers Labels
      TIERS.forEach(tier => {
        const lbl = document.getElementById('tier-lbl-' + tier.id);
        if (lbl) {
          const text = currentLang === 'zh' ? tier.label_zh : tier.label_en;
          lbl.innerHTML = `<span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:${tier.color}; box-shadow:0 0 8px ${tier.color}"></span> ${text}`;
        }
      });

      // Update Nodes Cards Content
      NODES.forEach(n => {
        const card = document.getElementById('node-' + n.id);
        if (card) {
          const title = currentLang === 'zh' ? n.title_zh : n.title_en;
          const desc = currentLang === 'zh' ? n.desc_zh : n.desc_en;
          const metricLbl = currentLang === 'zh' ? n.metricLabel_zh : n.metricLabel_en;
          const metricVal = currentLang === 'zh' ? n.metricVal_zh : n.metricVal_en;
          const tagTxt = t.tagMap[n.tag] || n.tag.toUpperCase();

          card.querySelector('.node-type-tag').innerText = tagTxt;
          card.querySelector('.node-title').innerText = title;
          card.querySelector('.node-desc').innerText = desc;
          card.querySelector('.metric-label').innerText = metricLbl;
          card.querySelector('.metric-val').innerText = metricVal;
        }
      });

      // Update Drawer if open
      if (selectedNodeObj) {
        selectNode(selectedNodeObj);
      }
    }

    // Render Architectural Tiers and Nodes
    function initArchitecture() {
      // 1. Render Tier Boxes
      TIERS.forEach(t => {
        const box = document.createElement('div');
        box.className = 'tier-box';
        box.style.left = t.x + 'px';
        box.style.top = t.y + 'px';
        box.style.width = t.w + 'px';
        box.style.height = t.h + 'px';
        box.style.borderColor = t.color + '44';

        const lbl = document.createElement('div');
        lbl.className = 'tier-label';
        lbl.id = 'tier-lbl-' + t.id;
        lbl.style.color = t.color;
        const text = currentLang === 'zh' ? t.label_zh : t.label_en;
        lbl.innerHTML = `<span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:${t.color}; box-shadow:0 0 8px ${t.color}"></span> ${text}`;
        box.appendChild(lbl);

        nodesLayer.appendChild(box);
      });

      // 2. Render Node Cards
      NODES.forEach(n => {
        const card = document.createElement('div');
        card.className = 'node-card';
        card.id = 'node-' + n.id;
        card.style.left = n.x + 'px';
        card.style.top = n.y + 'px';

        const title = currentLang === 'zh' ? n.title_zh : n.title_en;
        const desc = currentLang === 'zh' ? n.desc_zh : n.desc_en;
        const metricLbl = currentLang === 'zh' ? n.metricLabel_zh : n.metricLabel_en;
        const metricVal = currentLang === 'zh' ? n.metricVal_zh : n.metricVal_en;
        const tagTxt = I18N_UI[currentLang].tagMap[n.tag] || n.tag.toUpperCase();

        card.innerHTML = `
          <div class="node-header">
            <span class="node-type-tag tag-${n.tag}">${tagTxt}</span>
            <span style="font-size:10px; color:var(--text-muted);">●</span>
          </div>
          <div class="node-title">${title}</div>
          <div class="node-desc">${desc}</div>
          <div class="node-metric">
            <span class="metric-label">${metricLbl}</span>
            <span class="metric-val">${metricVal}</span>
          </div>
        `;

        card.addEventListener('click', (e) => {
          e.stopPropagation();
          selectNode(n);
        });

        nodesLayer.appendChild(card);
      });

      // 3. Render Connecting Wires
      drawConnections();
      updateTransform();
    }

    // Draw Smooth Bezier Connection Curves
    function drawConnections() {
      connLayer.innerHTML = '';
      CONNECTIONS.forEach(c => {
        const fromNode = NODES.find(n => n.id === c.from);
        const toNode = NODES.find(n => n.id === c.to);
        if (!fromNode || !toNode) return;

        const x1 = fromNode.x + 240;
        const y1 = fromNode.y + 60;
        const x2 = toNode.x;
        const y2 = toNode.y + 60;

        const dx = Math.abs(x2 - x1) * 0.55;
        const cp1x = x1 + Math.max(dx, 40);
        const cp1y = y1;
        const cp2x = x2 - Math.max(dx, 40);
        const cp2y = y2;

        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', `M ${x1} ${y1} C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${x2} ${y2}`);
        path.setAttribute('class', 'wire');
        path.id = `wire-${c.from}-${c.to}`;
        connLayer.appendChild(path);
      });
    }

    // Inspect Node Drawer
    function selectNode(n) {
      selectedNodeObj = n;
      document.querySelectorAll('.node-card').forEach(el => el.classList.remove('active'));
      const activeEl = document.getElementById('node-' + n.id);
      if (activeEl) activeEl.classList.add('active');

      document.querySelectorAll('path.wire').forEach(p => p.classList.remove('wire-active'));
      CONNECTIONS.forEach(c => {
        if (c.from === n.id || c.to === n.id) {
          const wire = document.getElementById(`wire-${c.from}-${c.to}`);
          if (wire) wire.classList.add('wire-active');
        }
      });

      const title = currentLang === 'zh' ? n.title_zh : n.title_en;
      const desc = currentLang === 'zh' ? n.desc_zh : n.desc_en;
      const tech = currentLang === 'zh' ? n.tech_zh : n.tech_en;
      const metrics = currentLang === 'zh' ? n.metrics_zh : n.metrics_en;
      const tagTxt = I18N_UI[currentLang].tagMap[n.tag] || n.tag.toUpperCase();

      document.getElementById('d-tag').className = `node-type-tag tag-${n.tag}`;
      document.getElementById('d-tag').innerText = tagTxt;
      document.getElementById('d-title').innerText = title;
      document.getElementById('d-path').innerText = n.path;
      document.getElementById('d-desc').innerText = desc;
      document.getElementById('d-tech').innerText = tech;

      const mg = document.getElementById('d-metrics');
      mg.innerHTML = '';
      if (metrics) {
        metrics.forEach(m => {
          const mCard = document.createElement('div');
          mCard.className = 'drawer-metric-card';
          mCard.innerHTML = `<div class="dm-val">${m.val}</div><div class="dm-lbl">${m.lbl}</div>`;
          mg.appendChild(mCard);
        });
      }

      document.getElementById('d-inputs').innerText = n.inputs || 'None';
      document.getElementById('d-outputs').innerText = n.outputs || 'None';

      document.getElementById('drawer').classList.add('open');
    }

    function closeDrawer() {
      selectedNodeObj = null;
      document.getElementById('drawer').classList.remove('open');
      document.querySelectorAll('.node-card').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('path.wire').forEach(p => p.classList.remove('wire-active'));
    }

    // Pan & Zoom Controls
    function updateTransform() {
      world.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
    }

    viewport.addEventListener('mousedown', (e) => {
      if (e.target.closest('#drawer') || e.target.closest('.controls-hud')) return;
      isDragging = true;
      startX = e.clientX - panX;
      startY = e.clientY - panY;
    });

    window.addEventListener('mousemove', (e) => {
      if (!isDragging) return;
      panX = e.clientX - startX;
      panY = e.clientY - startY;
      updateTransform();
    });

    window.addEventListener('mouseup', () => {
      isDragging = false;
    });

    viewport.addEventListener('wheel', (e) => {
      e.preventDefault();
      const zoomFactor = 1.08;
      const rect = viewport.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      let newScale = e.deltaY < 0 ? scale * zoomFactor : scale / zoomFactor;
      newScale = Math.min(Math.max(0.35, newScale), 2.2);

      panX = mouseX - (mouseX - panX) * (newScale / scale);
      panY = mouseY - (mouseY - panY) * (newScale / scale);
      scale = newScale;
      updateTransform();
    }, { passive: false });

    function zoomIn() {
      scale = Math.min(scale * 1.2, 2.2);
      updateTransform();
    }

    function zoomOut() {
      scale = Math.max(scale / 1.2, 0.35);
      updateTransform();
    }

    function resetView() {
      scale = 0.52;
      panX = 30;
      panY = 15;
      updateTransform();
      closeDrawer();
    }

    function toggleAnimation() {
      activeWireAnim = !activeWireAnim;
      const btn = document.getElementById('btn-anim');
      const t = I18N_UI[currentLang];
      document.querySelectorAll('path.wire').forEach(w => {
        w.style.animationPlayState = activeWireAnim ? 'running' : 'paused';
      });
      btn.innerHTML = activeWireAnim ? 
        `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg> <span>${t.btnAnimPause}</span>` :
        `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="10" y1="15" x2="10" y2="9"/><line x1="14" y1="15" x2="14" y2="9"/></svg> <span>${t.btnAnimResume}</span>`;
    }

    // Initialize on page load
    window.onload = () => {
      initArchitecture();
      applyLanguage();
      resetView();
    };
  </script>
</body>
</html>
'''
    # Write to docs/system_architecture.html
    docs_path = os.path.abspath('docs/system_architecture.html')
    with open(docs_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Generated: {docs_path}")

    # Write to submission/system_architecture.html
    sub_path = os.path.abspath('submission/system_architecture.html')
    with open(sub_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Generated: {sub_path}")

    # Update zip file
    zip_path = os.path.abspath('CDS540_RoadDamage_Submission_Pack.zip')
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'a', zipfile.ZIP_DEFLATED) as z:
            z.write('submission/system_architecture.html', 'submission/system_architecture.html')
            z.write('docs/system_architecture.html', 'docs/system_architecture.html')
        print(f"Updated ZIP: {zip_path} (Size: {round(os.path.getsize(zip_path)/(1024*1024), 2)} MB)")

if __name__ == '__main__':
    build_html()
