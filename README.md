# MaiChart AI - 舞萌自动写谱器

基于深度学习的 maimai 自动谱面生成系统，支持 Simai 格式输出。

## 项目结构

```
maimai/
├── src/
│   ├── parser/          # Simai 解析器
│   ├── audio/           # 音频特征提取
│   ├── data/            # 数据处理管道
│   ├── model/           # 模型架构
│   ├── training/        # 训练脚本
│   └── generation/      # 谱面生成
├── data/
│   ├── raw/             # 原始谱面文件
│   ├── processed/       # 处理后的数据
│   └── audio/           # 音频文件
├── models/              # 保存的模型
├── configs/             # 配置文件
├── notebooks/           # Jupyter notebooks
└── tests/               # 测试文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将 simai 谱面文件放入 `data/raw/`，音频文件放入 `data/audio/`。

### 3. 数据预处理

```bash
python -m src.data.preprocess
```

### 4. 训练模型

```bash
python -m src.training.train --max-epochs 100
```

### 5. 生成谱面

```bash
python -m src.generation.generate --audio your_audio.mp3 --output output.txt
```

## 技术架构

### 第一阶段：数据准备
- Simai 格式解析与规范化
- 自定义 Tokenizer 设计
- 音频-谱面对齐

### 第二阶段：特征提取
- 音频特征编码 (Mel-spectrogram, Onset, Beat)
- 预训练音频模型 embedding

### 第三阶段：模型架构
- Encoder-Decoder Transformer
- Cross-Attention 机制
- 多分类输出头

### 第四阶段：Simai 专项优化
- Slide 路径合法性验证
- BPM/时值转换
- 多押处理

### 第五阶段：闭环验证
- MajdataView 自动预览
- AstroDX 实机测试
- RLHF 人工反馈

## 支持的音符类型

- **TAP**: 普通点击 (1-8)
- **HOLD**: 长按 (1h[4:1])
- **SLIDE**: 滑动 (1-5[4:1])
- **TOUCH**: 触摸屏 (A1, B2, C, etc.)
- **BREAK**: 爆裂音符 (b前缀)
- **EX**: 特殊音符 (x前缀)

## 参考资料

- [Simai Wiki](https://w.atwiki.jp/simai)
- [Majdata 下载](https://moyingmoe.lanzouy.com/b03pbe8wj)
- MajdataView 播放器
- AstroDX 模拟器

## License

MIT License
