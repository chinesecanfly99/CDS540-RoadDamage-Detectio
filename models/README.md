# 模型权重说明 (Model Weights)

更新：2026-10-21。成员已在提交 `76c155b` 中将以下两份已有权重直接纳入 Git，并通过 `.gitignore` 例外保留。更新 `main` 或克隆当前版本即包含它们；本机已验证内容与此前权重完全一致。其他新增 checkpoint 仍默认忽略。

## 1. 权重文件清单与哈希校验

以下文件应随当前仓库出现在本目录（`models/`）下。若缺失，请先核对提交和仓库是否完整，不需要重复训练来生成它们：

| 模型文件 | 模型架构 | 文件大小 (Bytes) | SHA256 校验码 | 说明 |
| :--- | :--- | :---: | :--- | :--- |
| `baseline_best.pt` | YOLO11n | 21,389,027 | `b11873df6fe5d69e62612e6261d65598f3e5140a91f8c6a874941eca21e0c1ba` | 基线轻量检测模型 |
| `compare_yolo11s_best.pt` | YOLO11s | 19,203,601 | `cd6954c0d0e8f37fdc9f9dce91a1ec247632112ff8745bd5194f6ba9987c89a5` | 规模对照模型 |

### 校验命令：
- **Windows (PowerShell)**:
  ```powershell
  Get-FileHash models/baseline_best.pt -Algorithm SHA256
  Get-FileHash models/compare_yolo11s_best.pt -Algorithm SHA256
  ```
- **macOS / Linux**:
  ```bash
  shasum -a 256 models/baseline_best.pt
  shasum -a 256 models/compare_yolo11s_best.pt
  ```

## 2. 预训练骨干权重 (Ultralytics Pretrained)
若需从头微调或启动新实验，Ultralytics 会在首次运行时自动下载官方骨干预训练权重（如 `yolo11n.pt` 或 `yolo11s.pt`），默认保存在执行根目录下。
