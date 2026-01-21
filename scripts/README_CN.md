# 官方谱面导入工具

快速将官方谱面压缩包中的 simai 谱面和音频文件导入到项目的数据目录。

## 快速开始

### 基础使用

```bash
python scripts/import_official_charts.py
```

这将自动从 `D:\BaiduNetdiskDownload\官谱` 导入所有压缩包中的谱面和音频。

### 命令行选项

```
--source SOURCE          压缩包源目录（默认: D:\BaiduNetdiskDownload\官谱）
--raw-dir RAW_DIR        谱面输出目录（默认: data/raw）
--audio-dir AUDIO_DIR    音频输出目录（默认: data/audio）
--overwrite              覆盖已存在的文件
```

## 使用示例

### 1. 导入所有官方谱面（跳过已存在的文件）

```bash
python scripts/import_official_charts.py
```

输出示例：
```
2026-01-09 10:30:00 - INFO - 源目录: D:\BaiduNetdiskDownload\官谱
2026-01-09 10:30:00 - INFO - 谱面输出目录: d:\maimai\data\raw
2026-01-09 10:30:00 - INFO - 音频输出目录: d:\maimai\data\audio
2026-01-09 10:30:00 - INFO - 找到 12 个压缩包

处理: 01. maimai.zip
2026-01-09 10:30:05 - INFO - 发现 1 个版本
2026-01-09 10:30:15 - INFO - ✓ 导入谱面: Oshama Scramble
2026-01-09 10:30:25 - INFO - ✓ 导入音频: Oshama Scramble
...

==================================================
导入完成!
成功导入: 1234 个文件
跳过已存在: 45 个文件
当前谱面数: 1234
当前音频数: 1234
```

### 2. 使用自定义源目录

```bash
python scripts/import_official_charts.py --source "D:\my_charts"
```

### 3. 导出到自定义目录

```bash
python scripts/import_official_charts.py \
    --raw-dir "custom/charts" \
    --audio-dir "custom/audio"
```

### 4. 覆盖已存在的文件

```bash
python scripts/import_official_charts.py --overwrite
```

## 压缩包格式要求

脚本期望的压缩包结构：

```
版本名/
├── 歌曲名_1/
│   ├── maidata.txt        # Simai 格式谱面
│   └── track.mp3          # 音频文件
├── 歌曲名_2/
│   ├── maidata.txt
│   └── track.mp3
└── ...
```

## 文件命名规则

导入后的文件结构：

```
data/
├── raw/
│   ├── Oshama_Scramble.txt
│   ├── Love_You.txt
│   └── ...
└── audio/
    ├── Oshama_Scramble.mp3
    ├── Love_You.mp3
    └── ...
```

**重要**: 同一首歌的谱面和音频必须有相同的文件名（不包括扩展名）。

## 常见问题

### Q: 脚本很慢怎么办？

A: 这是正常的，因为需要从压缩包读取并写入大量音频文件（每个可能是 5-10MB）。
- 首次导入可能需要 30 分钟到 1 小时
- 可以打开任务管理器监查磁盘和内存占用

### Q: 可以中途停止吗？

A: 可以按 Ctrl+C 停止。脚本会跳过已导入的文件，下次运行时继续导入剩余的文件。

### Q: 如何检查导入结果？

A: 导入完成后，脚本会统计导入的谱面和音频数量。
或者手动查看：

```bash
# 查看谱面数量
Get-ChildItem data/raw -Filter "*.txt" | Measure-Object

# 查看音频数量
Get-ChildItem data/audio -Filter "*.mp3" | Measure-Object
```

### Q: 如何只导入某个版本？

A: 修改脚本，在 `get_zip_files()` 方法中筛选特定的压缩包。或者：

```bash
# 先解压到临时目录
Expand-Archive -Path "01. maimai.zip" -DestinationPath "temp"

# 然后手动复制文件到 data/raw 和 data/audio
```

## 导入后的下一步

1. **运行预处理**：
   ```bash
   python -m src.data.preprocess
   ```

2. **验证数据**：
   ```bash
   python -m src.data.dataset
   ```

3. **开始训练**：
   ```bash
   python -m src.training.train
   ```

详见 [完整使用指南](../docs/USAGE.md)。
