# 🎵 快速参考 - 官方谱面导入

## 最快开始 (3步)

### Step 1: 打开项目目录
```
cd d:\maimai
```

### Step 2: 运行导入脚本
**选择一种方式**：

```bash
# 方式 A: 双击运行（最简单）
scripts\import_charts.bat

# 方式 B: PowerShell
.\scripts\import_charts.ps1

# 方式 C: Python 命令行
python scripts/import_official_charts.py
```

### Step 3: 等待完成
- 首次导入需要 30 分钟 - 2 小时
- 监查进度日志中的 ✓ 标记

---

## 导入完成后

```bash
# 数据预处理
python -m src.data.preprocess

# 开始训练
python -m src.training.train --max-epochs 100
```

---

## 常用命令

```bash
# 查看谱面数量
(Get-ChildItem data/raw -Filter "*.txt").Count

# 查看音频数量  
(Get-ChildItem data/audio -Filter "*.mp3").Count

# 只导入特定版本
python scripts/import_official_charts.py \
    --source "D:\path\to\specific\version"

# 覆盖已有文件
python scripts/import_official_charts.py --overwrite

# 自定义输出目录
python scripts/import_official_charts.py \
    --raw-dir "custom/charts" \
    --audio-dir "custom/audio"
```

---

## 文件结构

```
d:\maimai\
├── scripts/
│   ├── import_official_charts.py    ⭐ 主脚本
│   ├── import_charts.bat             📝 快速启动 (Windows)
│   ├── import_charts.ps1             ⚡ 高级版本 (PowerShell)
│   ├── README_CN.md                  📖 详细文档
│   └── IMPLEMENTATION.md             🔧 技术细节
├── data/
│   ├── raw/                          📂 谱面输出位置
│   └── audio/                        📂 音频输出位置
└── docs/
    └── USAGE.md                      📚 已更新说明
```

---

## 压缩包位置

```
D:\BaiduNetdiskDownload\官谱\
├── 01. maimai.zip
├── 02. maimai PLUS.zip
├── 02.5. 舞萌中国.zip
├── 03. GreeN.zip
├── 04. GreeN PLUS.zip
└── ... (12+ 个版本)
```

---

## 预期结果

✅ **导入完成后**:
- `data/raw/` 中有 1000+ 个 `.txt` 谱面
- `data/audio/` 中有 1000+ 个 `.mp3` 音频
- 文件名完全匹配（同一首歌）
- 可直接用于模型训练

---

## 遇到问题？

| 问题 | 解决方案 |
|------|---------|
| 找不到 zip | 检查 `D:\BaiduNetdiskDownload\官谱` 目录 |
| 导入很慢 | 正常（音频文件大），耐心等待 |
| 导入失败 | 用 `--overwrite` 重新导入，或检查磁盘空间 |
| 文件不完整 | 按 Ctrl+C 停止，重新运行继续导入 |

---

## 详细文档

- **使用指南**: [scripts/README_CN.md](README_CN.md)
- **技术细节**: [scripts/IMPLEMENTATION.md](IMPLEMENTATION.md)  
- **项目文档**: [docs/USAGE.md](../docs/USAGE.md)

---

**提示**: 将此文件保存为书签，方便后续查阅！
