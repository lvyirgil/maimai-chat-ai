# 官方谱面导入 - 实现总结

## 📦 创建的文件

### 1. **Python 导入脚本** - `scripts/import_official_charts.py`
   - 完整的谱面和音频导入工具
   - 支持从多个压缩包自动提取
   - 自动处理文件名匹配
   - 详细的日志输出
   - 支持跳过已存在的文件或覆盖

**主要功能**:
- ✅ 自动从 `D:\BaiduNetdiskDownload\官谱` 找到所有 zip 文件
- ✅ 解析压缩包结构（版本/歌曲/文件）
- ✅ 提取 `maidata.txt`（simai 谱面）到 `data/raw/`
- ✅ 提取 `track.mp3`（音频）到 `data/audio/`
- ✅ 自动文件名匹配和去重
- ✅ 详细的进度日志和错误处理

### 2. **快速启动脚本** - `scripts/import_charts.bat`（Windows 批处理）
   - 简单的双击运行脚本
   - 自动检查项目和压缩包目录
   - 友好的用户提示

### 3. **PowerShell 脚本** - `scripts/import_charts.ps1`
   - 高级功能版本
   - 支持命令行参数
   - 彩色输出和确认提示
   - 专业的日志输出

### 4. **使用文档** - `scripts/README_CN.md`
   - 详细的使用指南
   - 命令行选项说明
   - 使用示例
   - 常见问题解答

### 5. **主文档更新** - `docs/USAGE.md`
   - 在"数据收集"部分添加了官方谱面导入说明
   - 包含快速命令和压缩包格式要求

---

## 🚀 使用方式

### 方式 1: 双击运行（最简单）
```
scripts/import_charts.bat
```

### 方式 2: PowerShell（推荐）
```powershell
# 基础使用
.\scripts\import_charts.ps1

# 自定义源目录
.\scripts\import_charts.ps1 -Source "D:\my_charts"

# 覆盖已存在的文件
.\scripts\import_charts.ps1 -Overwrite
```

### 方式 3: Python 直接运行（最灵活）
```bash
# 默认设置
python scripts/import_official_charts.py

# 自定义选项
python scripts/import_official_charts.py \
    --source "D:\charts" \
    --raw-dir "data/raw" \
    --audio-dir "data/audio" \
    --overwrite
```

---

## 📋 压缩包格式要求

脚本自动处理以下格式：

```
D:\BaiduNetdiskDownload\官谱\
├── 01. maimai.zip
│   └── 01. maimai/
│       ├── Song_Name_1/
│       │   ├── maidata.txt      (Simai格式谱面)
│       │   ├── track.mp3        (音频)
│       │   ├── bg.jpg           (背景图，忽略)
│       │   └── pv.mp4           (视频，忽略)
│       └── Song_Name_2/
│           └── ...
├── 02. maimai PLUS.zip
│   └── 02. maimai PLUS/
│       └── ...
└── ...其他版本...
```

---

## 📊 导入后的数据结构

```
data/
├── raw/                              # 谱面目录
│   ├── Oshama_Scramble.txt
│   ├── Love_You.txt
│   ├── City_Escape_Act1.txt
│   └── ...（所有谱面）
└── audio/                            # 音频目录
    ├── Oshama_Scramble.mp3
    ├── Love_You.mp3
    ├── City_Escape_Act1.mp3
    └── ...（所有音频）
```

**重要**: 同一首歌的 `.txt` 和 `.mp3` 必须有相同的文件名。

---

## ⚙️ 命令行选项

```
--source SOURCE          压缩包源目录
                         (默认: D:\BaiduNetdiskDownload\官谱)

--raw-dir RAW_DIR        谱面输出目录  
                         (默认: data/raw)

--audio-dir AUDIO_DIR    音频输出目录
                         (默认: data/audio)

--overwrite              覆盖已存在的文件
                         (默认: 跳过已存在的文件)
```

---

## 📈 预期性能

- **首次导入**: 取决于压缩包大小
  - 单个版本 (~10-20GB): 10-20 分钟
  - 所有版本 (~60GB+): 1-2 小时
  
- **后续导入**: 只导入新文件，速度更快
  - 由于有跳过机制，只处理新增的文件

---

## ✅ 检查导入结果

### PowerShell
```powershell
# 查看导入的谱面数量
(Get-ChildItem data/raw -Filter "*.txt").Count

# 查看导入的音频数量
(Get-ChildItem data/audio -Filter "*.mp3").Count

# 列出前 10 个导入的文件
Get-ChildItem data/raw -Filter "*.txt" | Select-Object Name | Head -10
```

### 命令行
```bash
# Windows CMD
dir /b data\raw\*.txt | find /c ".txt"
dir /b data\audio\*.mp3 | find /c ".mp3"
```

---

## 🔧 下一步

导入完成后，进行数据预处理：

```bash
# 预处理所有数据
python -m src.data.preprocess

# 使用自定义参数
python -m src.data.preprocess \
    --raw-dir data/raw \
    --audio-dir data/audio \
    --output-dir data/processed \
    --max-seq-length 4096
```

然后开始训练：

```bash
python -m src.training.train --max-epochs 100
```

详见 [完整使用指南](../docs/USAGE.md)。

---

## 🐛 常见问题

### 1. 脚本运行很慢？
- 这是正常的，音频文件很大（每个 2-10MB）
- 磁盘 I/O 是主要瓶颈
- 可以在任务管理器监查进度

### 2. 中途出错了怎么办？
- 脚本会跳过已导入的文件
- 修复问题后重新运行即可
- 使用 `--overwrite` 重新导入失败的文件

### 3. 如何只导入特定版本？
修改脚本的 `get_zip_files()` 方法，或手动操作：

```powershell
# 解压单个版本
Expand-Archive -Path "01. maimai.zip" -DestinationPath "temp_extract"

# 手动复制文件到 data/raw 和 data/audio
```

### 4. 文件名有中文/特殊字符怎么办？
- 脚本自动处理 UTF-8 编码
- 中文文件名会被保留

### 5. 需要多少磁盘空间？
- 所有压缩包解压后 ~100-150GB
- 建议保留 200GB+ 的磁盘空间
- 可以在导入完成后删除压缩包

---

## 📝 文件清单

| 文件 | 说明 | 用途 |
|------|------|------|
| `scripts/import_official_charts.py` | 主导入脚本 | 核心功能 |
| `scripts/import_charts.bat` | Windows 批处理 | 快速启动 |
| `scripts/import_charts.ps1` | PowerShell 脚本 | 高级功能 |
| `scripts/README_CN.md` | 详细文档 | 参考指南 |
| `docs/USAGE.md` | 主文档（已更新） | 集成文档 |

---

## 💡 技术细节

### 脚本工作流程

1. **初始化**: 创建输出目录，设置日志
2. **扫描**: 找到所有 zip 文件
3. **解压**: 使用 Python 的 `zipfile` 模块流式解压
4. **处理**: 
   - 识别版本文件夹
   - 找到 `maidata.txt` 和 `track.mp3`
   - 检查文件是否已存在
   - 写入到目标目录
5. **统计**: 生成导入报告

### 关键特性

- **流式处理**: 不需要完整解压到磁盘
- **自动去重**: 检查同名文件防止覆盖
- **错误恢复**: 失败文件不影响其他文件
- **详细日志**: 便于调试和监查进度

---

**创建时间**: 2026-01-09
**版本**: 1.0
