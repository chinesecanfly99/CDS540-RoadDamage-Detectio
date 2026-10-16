# 模型权重说明 (Model Weights)

由于 GitHub 仓库推荐仅保留轻量代码和文档，大于 10MB 的模型权重文件已加入 `.gitignore`，不随代码库直接上传。

## 1. 权重文件清单与哈希校验

请从小组成员共享网盘或 Release 中下载以下两份已有权重，并直接存放在本目录（`models/`）下：

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
