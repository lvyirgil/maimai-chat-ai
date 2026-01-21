# 📥 官方谱面导入工具 - 已安装

## ✅ 已创建文件

```
scripts/
├── import_official_charts.py  (7.9 KB)  ⭐ 核心导入脚本
├── import_charts.bat          (1.4 KB)  📝 Windows 快速启动
├── import_charts.ps1          (3.1 KB)  ⚡ PowerShell 脚本
├── README_CN.md               (3.9 KB)  📖 详细使用文档
├── QUICK_START.md             (3.0 KB)  🚀 快速参考
└── IMPLEMENTATION.md          (6.9 KB)  🔧 技术细节说明
```

## 🚀 立即开始

### 最简单的方式（推荐）

```bash
# 方式 1: 双击运行
scripts\import_charts.bat

# 方式 2: PowerShell
.\scripts\import_charts.ps1

# 方式 3: 命令行
python scripts/import_official_charts.py
```

## 📊 工作原理

1. **自动扫描** `D:\BaiduNetdiskDownload\官谱` 中的所有 `.zip` 文件
2. **智能解析** 压缩包内的文件结构
3. **提取文件**:
   - `maidata.txt` (simai 谱面) → `data/raw/`
   - `track.mp3` (音频) → `data/audio/`
4. **自动去重** 检查并跳过已存在的文件
5. **生成报告** 显示导入统计信息

## ⏱️ 预期耗时

- 首次导入所有官方谱面: **1-2 小时**（取决于硬件）
- 之后增量导入: **更快** (仅处理新文件)

## 📚 文档

| 文档 | 用途 |
|------|------|
| [scripts/QUICK_START.md](scripts/QUICK_START.md) | ⚡ 快速参考卡片 |
| [scripts/README_CN.md](scripts/README_CN.md) | 📖 详细使用文档 |
| [scripts/IMPLEMENTATION.md](scripts/IMPLEMENTATION.md) | 🔧 技术实现细节 |
| [docs/USAGE.md](docs/USAGE.md) | 📚 项目完整指南 |

## 🔧 常用命令

```bash
# 基础使用
python scripts/import_official_charts.py

# 自定义源目录
python scripts/import_official_charts.py --source "D:\my_charts"

# 覆盖已存在的文件
python scripts/import_official_charts.py --overwrite

# 查看帮助
python scripts/import_official_charts.py --help
```

## 📋 运行要求

- ✅ Python 3.7+（已有）
- ✅ 压缩包在 `D:\BaiduNetdiskDownload\官谱`（需要）
- ✅ ~150GB 可用磁盘空间（用于解压）
- ✅ 10+ 分钟耐心（首次导入）

## ✨ 特点

- 🎯 **自动化**: 无需手动解压
- 🔄 **增量导入**: 跳过已导入的文件
- 📝 **详细日志**: 完整的进度信息
- 🛡️ **容错**: 失败文件不影响其他文件
- 💨 **快速**: 流式处理，不占用大量内存

## 🎬 导入后的下一步

```bash
# 1. 运行数据预处理
python -m src.data.preprocess

# 2. 开始模型训练
python -m src.training.train --max-epochs 100
```

---

**创建时间**: 2026-01-09  
**版本**: 1.0  
**状态**: ✅ 就绪可用
