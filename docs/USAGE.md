# MaiChart AI 完整使用指南

## 目录

1. [环境准备](#环境准备)
2. [数据收集](#数据收集)
3. [数据预处理](#数据预处理)
4. [模型训练](#模型训练)
5. [谱面生成](#谱面生成)
6. [导出与预览](#导出与预览)
7. [高级配置](#高级配置)

---

## 环境准备

### 系统要求

- Python 3.9+
- CUDA 11.7+ (用于 GPU 训练，可选)
- 8GB+ RAM
- 50GB+ 磁盘空间（用于数据和模型）

### 安装步骤

```bash
# 1. 克隆项目
cd d:\maimai

# 2. 创建虚拟环境（推荐）
python -m venv venv
.\venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装 PyTorch（根据你的 CUDA 版本）
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 仅 CPU
pip install torch torchvision torchaudio
```

### 验证安装

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

import librosa
print(f"Librosa: {librosa.__version__}")
```

---

## 数据收集

### 谱面来源

1. **社区谱面**: 从 maimai 同人社区收集
2. **官方转换**: 使用工具将官方谱面转换为 simai 格式
3. **官方谱面导入**: 从官方谱面压缩包导入（推荐）

### 快速导入官方谱面

如果你有官方谱面压缩包（位于 `D:\BaiduNetdiskDownload\官谱`），可以使用导入脚本自动提取谱面和音频：

```bash
# 导入所有官方谱面
python scripts/import_official_charts.py

# 自定义源目录
python scripts/import_official_charts.py --source "D:\path\to\charts"

# 自定义输出目录
python scripts/import_official_charts.py --raw-dir "data/raw" --audio-dir "data/audio"

# 覆盖已存在的文件
python scripts/import_official_charts.py --overwrite
```

**压缩包结构要求**:
```
官谱/
├── 01. maimai.zip
│   └── 01. maimai/
│       ├── Song_Name_1/
│       │   ├── maidata.txt (Simai格式)
│       │   └── track.mp3 (音频)
│       └── Song_Name_2/
│           └── ...
├── 02. maimai PLUS.zip
└── ...
```

### 目录结构

```
data/
├── raw/
│   ├── Song_Name_1.txt    # simai 格式谱面
│   ├── Song_Name_2.txt
│   └── ...
└── audio/
    ├── Song_Name_1.mp3    # 对应音频（必须同名）
    ├── Song_Name_2.mp3
    └── ...
```

### 推荐数据量

| 阶段 | 数据量 | 预期效果 |
|------|--------|----------|
| 测试 | 10-50 首 | 验证流程 |
| 初步 | 100-500 首 | 基本可用 |
| 理想 | 500-2000 首 | 良好效果 |

---

## 数据预处理

### 运行预处理

```bash
python -m src.data.preprocess \
    --raw-dir data/raw \
    --audio-dir data/audio \
    --output-dir data/processed \
    --max-seq-length 4096
```

### 预处理输出

```
data/processed/
├── config.json          # 配置信息
├── sample_00000.npz     # 处理后的样本
├── sample_00001.npz
└── ...
```

### 检查数据

```python
from src.data import create_dataset, DatasetConfig

config = DatasetConfig(processed_dir="data/processed")
dataset = create_dataset(config)
dataset.load()

print(f"样本数量: {len(dataset)}")
sample = dataset[0]
print(f"音频特征形状: {sample.audio_features.shape}")
print(f"谱面 token 数: {sample.chart_tokens.shape}")
```

---

## 模型训练

### 基本训练

```bash
python -m src.training.train \
    --data-dir data/processed \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --max-epochs 100 \
    --save-dir models
```

### 使用 GPU 训练

确保 CUDA 可用，脚本会自动使用 GPU。

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch-size` | 8 | 批大小，根据显存调整 |
| `--learning-rate` | 1e-4 | 学习率 |
| `--max-epochs` | 100 | 最大训练轮数 |
| `--warmup-steps` | 1000 | 学习率预热步数 |
| `--hidden-dim` | 512 | 模型隐藏层维度 |
| `--n-heads` | 8 | 注意力头数 |
| `--n-layers` | 6 | Transformer 层数 |

### 监控训练

```bash
# 使用 TensorBoard
tensorboard --logdir logs

# 或使用 WandB
python -m src.training.train --use-wandb
```

### 恢复训练

```bash
python -m src.training.train --resume models/checkpoint.pt
```

---

## 谱面生成

### 基本生成

```bash
python -m src.generation.generate \
    --audio your_song.mp3 \
    --model models/best.pt \
    --output generated_chart.txt
```

### 自定义元数据

```bash
python -m src.generation.generate \
    --audio your_song.mp3 \
    --model models/best.pt \
    --output generated_chart.txt \
    --title "歌曲名称" \
    --bpm 150 \
    --level "12+" \
    --designer "AI"
```

### 调整生成参数

```bash
python -m src.generation.generate \
    --audio your_song.mp3 \
    --model models/best.pt \
    --temperature 0.8 \     # 越高越随机
    --top-k 50 \            # Top-K 采样
    --top-p 0.9             # Nucleus 采样
```

### Python API

```python
from src.generation import ChartGenerator, GenerationConfig

# 加载模型
generator = ChartGenerator("models/best.pt")

# 配置
config = GenerationConfig(
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    validate_slides=True,
    check_playability=True
)

# 生成
simai_text = generator.generate("your_song.mp3", config)
print(simai_text)
```

---

## 导出与预览

### 导出到 Majdata

```bash
python -m src.generation.export \
    --chart generated_chart.txt \
    --audio your_song.mp3 \
    --output charts/output \
    --name "歌曲名称" \
    --format majdata
```

### 导出到 AstroDX

```bash
python -m src.generation.export \
    --chart generated_chart.txt \
    --audio your_song.mp3 \
    --output charts/output \
    --format astrodx
```

### 自动预览

```bash
python -m src.generation.export \
    --chart generated_chart.txt \
    --audio your_song.mp3 \
    --output charts/output \
    --preview \
    --majdata-path "C:\Path\To\Majdata"
```

### Python API

```python
from src.generation import export_for_majdata, MajdataExporter

# 简单导出
export_for_majdata(
    simai_text=simai_text,
    audio_path="your_song.mp3",
    output_dir="charts/output",
    song_name="歌曲名称"
)

# 带预览
config = MajdataConfig(majdata_path="C:\\Path\\To\\Majdata")
exporter = MajdataExporter(config)
output = exporter.export(simai_text, "your_song.mp3", "charts/output")
exporter.launch_preview(output)
```

---

## 高级配置

### 模型配置

编辑 `configs/default.yaml`:

```yaml
model:
  audio_encoder:
    type: "transformer"
    d_model: 512       # 增加维度提升容量
    n_heads: 8
    n_layers: 6        # 增加层数
    dropout: 0.1
  
  chart_decoder:
    type: "transformer"
    d_model: 512
    n_heads: 8
    n_layers: 8
    dropout: 0.1
```

### 数据增强

在训练时可以应用数据增强:

```python
# 时间偏移
# BPM 扰动
# 音高转换
```

### 后处理规则

编辑 `src/generation/generate.py` 中的 `PostProcessor`:

- 密度调整
- BREAK 添加规则
- Slide 简化

---

## 常见问题

### Q: 训练时显存不足

A: 减小 `batch_size` 或 `hidden_dim`

### Q: 生成的谱面不太好听

A: 
1. 增加训练数据量
2. 调整 temperature (降低使输出更保守)
3. 使用更高质量的训练数据

### Q: 解析谱面失败

A: 确保谱面是标准 simai 格式，可以在 MajdataView 中正常播放

### Q: 音频特征提取失败

A: 确保音频格式为 MP3/WAV/OGG，且可以正常播放

---

## 技术细节

### 模型架构

```
Audio Features (Mel + Onset + Chroma)
         ↓
┌─────────────────────┐
│   Audio Encoder     │
│   (Transformer)     │
└─────────────────────┘
         ↓
    Memory (encoded)
         ↓
┌─────────────────────┐
│   Chart Decoder     │
│   (Transformer)     │  ← Previous tokens
└─────────────────────┘
         ↓
    Token Logits
```

### Token 设计

- 位置 token: 1-8, A1-E8, C
- 动作 token: TAP, HOLD, SLIDE, TOUCH
- 修饰符 token: BREAK, EX, STAR
- 时值 token: 基于 1/192 分音符
- 控制 token: BOS, EOS, SEP, EACH

### 训练技巧

1. 使用 Label Smoothing
2. 应用 Warmup
3. 梯度裁剪
4. 早停

---

## 参考链接

- [Simai 官方 Wiki](https://w.atwiki.jp/simai)
- [Majdata 下载](https://moyingmoe.lanzouy.com/b03pbe8wj)
- [Librosa 文档](https://librosa.org/doc/)
- [PyTorch 文档](https://pytorch.org/docs/)
