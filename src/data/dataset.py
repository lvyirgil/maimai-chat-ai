"""
数据集处理模块
将原始谱面和音频数据转换为训练数据
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Generator
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from ..parser.simai_parser import parse_simai, Chart
from ..audio.audio_features import AudioFeatureExtractor, AudioFeatures
from .tokenizer import SimaiTokenizer, TokenizedChart


@dataclass
class ChartSample:
    """单个谱面样本"""
    # 元数据
    song_id: str
    difficulty: int
    bpm: float
    
    # 音频特征
    audio_features: np.ndarray  # (n_frames, feature_dim)
    
    # 谱面 token
    chart_tokens: np.ndarray    # (seq_length,)
    attention_mask: np.ndarray  # (seq_length,)
    
    # 时间对齐信息
    frame_to_token_map: np.ndarray  # (n_frames,) -> token index


@dataclass
class DatasetConfig:
    """数据集配置"""
    raw_dir: str = "data/raw"
    audio_dir: str = "data/audio"
    processed_dir: str = "data/processed"
    
    # 音频配置
    sample_rate: int = 22050
    hop_length: int = 512
    n_mels: int = 128
    
    # Tokenizer 配置
    max_seq_length: int = 4096
    
    # 数据划分
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1


class ChartDataset:
    """谱面数据集"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.tokenizer = SimaiTokenizer(config.max_seq_length)
        self.audio_extractor = AudioFeatureExtractor(
            sample_rate=config.sample_rate,
            hop_length=config.hop_length,
            n_mels=config.n_mels
        )
        
        self.samples: List[ChartSample] = []
        
    def load_raw_data(self) -> List[Tuple[str, Chart, str]]:
        """加载原始数据"""
        raw_dir = Path(self.config.raw_dir)
        audio_dir = Path(self.config.audio_dir)
        
        data = []
        
        # 遍历谱面文件
        for chart_file in raw_dir.glob("**/*.txt"):
            song_id = chart_file.stem
            
            # 查找对应音频
            audio_file = None
            for ext in ['.mp3', '.wav', '.ogg', '.flac']:
                candidate = audio_dir / f"{song_id}{ext}"
                if candidate.exists():
                    audio_file = candidate
                    break
            
            if audio_file is None:
                print(f"警告: 找不到 {song_id} 的音频文件")
                continue
            
            # 解析谱面
            try:
                with open(chart_file, 'r', encoding='utf-8') as f:
                    simai_text = f.read()
                chart = parse_simai(simai_text)
                data.append((song_id, chart, str(audio_file)))
            except Exception as e:
                print(f"解析 {chart_file} 失败: {e}")
                continue
        
        return data
    
    def process_sample(self, song_id: str, chart: Chart, audio_path: str) -> Optional[ChartSample]:
        """处理单个样本"""
        try:
            # 提取音频特征
            audio_features = self.audio_extractor.extract(audio_path)
            
            # Tokenize 谱面
            tokenized = self.tokenizer.tokenize(chart)
            
            # 构建时间对齐
            frame_to_token_map = self._build_alignment(
                audio_features.frame_times,
                tokenized,
                chart.meta.bpm
            )
            
            # 合并音频特征
            combined_features = np.concatenate([
                audio_features.mel_spectrogram.T,  # (n_frames, n_mels)
                audio_features.onset_strength.reshape(-1, 1),  # (n_frames, 1)
                audio_features.chroma.T,  # (n_frames, 12)
            ], axis=1)
            
            return ChartSample(
                song_id=song_id,
                difficulty=chart.meta.difficulty,
                bpm=chart.meta.bpm,
                audio_features=combined_features,
                chart_tokens=tokenized.token_ids,
                attention_mask=tokenized.attention_mask,
                frame_to_token_map=frame_to_token_map
            )
        except Exception as e:
            print(f"处理 {song_id} 失败: {e}")
            return None
    
    def _build_alignment(
        self,
        frame_times: np.ndarray,
        tokenized: TokenizedChart,
        bpm: float
    ) -> np.ndarray:
        """构建音频帧到 token 的对齐映射"""
        beat_duration = 60.0 / bpm
        tick_duration = beat_duration / 48
        
        # 计算每个帧对应的 token 索引
        token_indices = np.zeros(len(frame_times), dtype=np.int32)
        
        current_token_idx = 0
        current_time = 0.0
        
        for frame_idx, frame_time in enumerate(frame_times):
            # 找到最近的 token
            while current_token_idx < len(tokenized.tokens) - 1:
                # SEP token 推进时间
                if tokenized.tokens[current_token_idx].type.name == 'SEP':
                    current_time += tick_duration
                
                if current_time > frame_time:
                    break
                current_token_idx += 1
            
            token_indices[frame_idx] = current_token_idx
        
        return token_indices
    
    def process_all(self, save: bool = True) -> None:
        """处理所有数据"""
        raw_data = self.load_raw_data()
        print(f"找到 {len(raw_data)} 个谱面")
        
        self.samples = []
        for song_id, chart, audio_path in tqdm(raw_data, desc="处理谱面"):
            sample = self.process_sample(song_id, chart, audio_path)
            if sample:
                self.samples.append(sample)
        
        print(f"成功处理 {len(self.samples)} 个样本")
        
        if save:
            self.save()
    
    def save(self, path: Optional[str] = None) -> None:
        """保存处理后的数据"""
        save_dir = Path(path or self.config.processed_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        with open(save_dir / "config.json", 'w') as f:
            json.dump({
                'sample_rate': self.config.sample_rate,
                'hop_length': self.config.hop_length,
                'n_mels': self.config.n_mels,
                'max_seq_length': self.config.max_seq_length,
                'num_samples': len(self.samples),
            }, f, indent=2)
        
        # 保存样本
        for i, sample in enumerate(tqdm(self.samples, desc="保存样本")):
            sample_path = save_dir / f"sample_{i:05d}.npz"
            np.savez_compressed(
                sample_path,
                song_id=sample.song_id,
                difficulty=sample.difficulty,
                bpm=sample.bpm,
                audio_features=sample.audio_features,
                chart_tokens=sample.chart_tokens,
                attention_mask=sample.attention_mask,
                frame_to_token_map=sample.frame_to_token_map
            )
        
        print(f"数据已保存到 {save_dir}")
    
    def load(self, path: Optional[str] = None) -> None:
        """加载处理后的数据"""
        load_dir = Path(path or self.config.processed_dir)
        
        # 加载配置
        with open(load_dir / "config.json", 'r') as f:
            config = json.load(f)
        
        # 加载样本
        self.samples = []
        sample_files = sorted(load_dir.glob("sample_*.npz"))
        
        for sample_path in tqdm(sample_files, desc="加载样本"):
            data = np.load(sample_path, allow_pickle=True)
            sample = ChartSample(
                song_id=str(data['song_id']),
                difficulty=int(data['difficulty']),
                bpm=float(data['bpm']),
                audio_features=data['audio_features'],
                chart_tokens=data['chart_tokens'],
                attention_mask=data['attention_mask'],
                frame_to_token_map=data['frame_to_token_map']
            )
            self.samples.append(sample)
        
        print(f"加载了 {len(self.samples)} 个样本")
    
    def split(self) -> Tuple['ChartDataset', 'ChartDataset', 'ChartDataset']:
        """划分训练/验证/测试集"""
        np.random.shuffle(self.samples)
        
        n = len(self.samples)
        train_end = int(n * self.config.train_split)
        val_end = train_end + int(n * self.config.val_split)
        
        train_dataset = ChartDataset(self.config)
        train_dataset.samples = self.samples[:train_end]
        train_dataset.tokenizer = self.tokenizer
        
        val_dataset = ChartDataset(self.config)
        val_dataset.samples = self.samples[train_end:val_end]
        val_dataset.tokenizer = self.tokenizer
        
        test_dataset = ChartDataset(self.config)
        test_dataset.samples = self.samples[val_end:]
        test_dataset.tokenizer = self.tokenizer
        
        return train_dataset, val_dataset, test_dataset
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> ChartSample:
        return self.samples[idx]
    
    def batch_generator(
        self,
        batch_size: int,
        shuffle: bool = True
    ) -> Generator[Dict[str, np.ndarray], None, None]:
        """生成批次数据"""
        indices = np.arange(len(self.samples))
        if shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_samples = [self.samples[idx] for idx in batch_indices]
            
            # 获取最大序列长度
            max_audio_len = max(s.audio_features.shape[0] for s in batch_samples)
            max_token_len = self.config.max_seq_length
            
            # 构建批次张量
            batch_audio = np.zeros((len(batch_samples), max_audio_len, batch_samples[0].audio_features.shape[1]))
            batch_tokens = np.zeros((len(batch_samples), max_token_len), dtype=np.int32)
            batch_mask = np.zeros((len(batch_samples), max_token_len), dtype=np.int32)
            batch_audio_mask = np.zeros((len(batch_samples), max_audio_len), dtype=np.int32)
            
            for j, sample in enumerate(batch_samples):
                audio_len = sample.audio_features.shape[0]
                batch_audio[j, :audio_len] = sample.audio_features
                batch_tokens[j] = sample.chart_tokens
                batch_mask[j] = sample.attention_mask
                batch_audio_mask[j, :audio_len] = 1
            
            yield {
                'audio_features': batch_audio,
                'audio_mask': batch_audio_mask,
                'chart_tokens': batch_tokens,
                'attention_mask': batch_mask,
            }


def create_dataset(config: Optional[DatasetConfig] = None) -> ChartDataset:
    """创建数据集"""
    if config is None:
        config = DatasetConfig()
    return ChartDataset(config)


if __name__ == "__main__":
    # 测试
    config = DatasetConfig()
    dataset = create_dataset(config)
    
    # 如果有数据，处理它
    raw_data = dataset.load_raw_data()
    if raw_data:
        print(f"找到 {len(raw_data)} 个谱面，开始处理...")
        dataset.process_all(save=True)
    else:
        print("请将谱面文件放入 data/raw/，音频文件放入 data/audio/")
