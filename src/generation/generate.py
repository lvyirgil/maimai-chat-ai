"""
谱面生成模块
从音频生成 simai 格式谱面
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import numpy as np
import torch

from ..model.transformer import MaiChartModel, ModelConfig, create_model
from ..audio.audio_features import AudioFeatureExtractor, AudioFeatures
from ..data.tokenizer import SimaiTokenizer
from ..parser.simai_parser import (
    Note, NoteType, ChartMeta, Chart,
    SimaiGenerator, generate_simai
)


@dataclass
class GenerationConfig:
    """生成配置"""
    # 采样参数
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    
    # 生成限制
    max_length: int = 4096
    
    # 后处理
    validate_slides: bool = True
    check_playability: bool = True
    simplify_duration: bool = True
    
    # 输出
    difficulty: int = 5  # 默认 Master 难度


class ChartGenerator:
    """谱面生成器"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        
        # 加载模型
        self.model, self.model_config = self._load_model(model_path)
        self.model.eval()
        
        # 初始化工具
        self.audio_extractor = AudioFeatureExtractor()
        self.tokenizer = SimaiTokenizer()
    
    def _load_model(self, model_path: str) -> tuple:
        """加载训练好的模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 恢复配置
        model_config = ModelConfig(**checkpoint['model_config'])
        
        # 创建模型
        model = create_model(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        return model, model_config
    
    @torch.no_grad()
    def generate(
        self,
        audio_path: str,
        config: Optional[GenerationConfig] = None,
        meta: Optional[ChartMeta] = None
    ) -> str:
        """
        从音频生成谱面
        
        Args:
            audio_path: 音频文件路径
            config: 生成配置
            meta: 谱面元数据
        
        Returns:
            simai 格式的谱面文本
        """
        if config is None:
            config = GenerationConfig()
        
        # 提取音频特征
        print(f"提取音频特征: {audio_path}")
        audio_features = self.audio_extractor.extract(audio_path)
        
        # 准备元数据
        if meta is None:
            meta = ChartMeta(
                title=Path(audio_path).stem,
                bpm=audio_features.tempo,
                offset=0.0,
                level="?",
                designer="AI",
                difficulty=config.difficulty
            )
        else:
            # 如果未指定 BPM，使用检测到的
            if meta.bpm == 0:
                meta.bpm = audio_features.tempo
        
        # 准备模型输入
        combined_features = np.concatenate([
            audio_features.mel_spectrogram.T,
            audio_features.onset_strength.reshape(-1, 1),
            audio_features.chroma.T,
        ], axis=1)
        
        audio_tensor = torch.tensor(
            combined_features,
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        
        audio_mask = torch.ones(
            1, audio_tensor.size(1),
            dtype=torch.float32
        ).to(self.device)
        
        # 生成 token 序列
        print("生成谱面...")
        generated_tokens = self.model.generate(
            audio_features=audio_tensor,
            audio_mask=audio_mask,
            max_length=config.max_length,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            bos_token_id=1,
            eos_token_id=2
        )
        
        # 解码为音符
        token_ids = generated_tokens[0].cpu().numpy()
        notes = self.tokenizer.detokenize(token_ids, bpm=meta.bpm)
        
        print(f"生成了 {len(notes)} 个音符")
        
        # 后处理
        if config.validate_slides:
            notes = self._validate_slides(notes)
        
        if config.check_playability:
            notes = self._check_playability(notes)
        
        # 生成 simai 文本
        simai_text = self._notes_to_simai(notes, meta)
        
        return simai_text
    
    def _validate_slides(self, notes: List[Note]) -> List[Note]:
        """验证和修复 Slide 路径"""
        valid_notes = []
        
        for note in notes:
            if note.note_type == NoteType.SLIDE:
                # 检查起点和终点是否有效
                if note.position and note.slide_end:
                    start = int(note.position)
                    end = int(note.slide_end)
                    
                    # 基本验证：起点终点在 1-8 范围内
                    if 1 <= start <= 8 and 1 <= end <= 8:
                        # 检查路径类型是否合理
                        if note.slide_path:
                            # 对于某些路径类型，检查是否可达
                            # 简化：假设所有路径都有效
                            valid_notes.append(note)
                        else:
                            # 没有路径，添加默认直线路径
                            note.slide_path = f"{start}-{end}"
                            valid_notes.append(note)
                else:
                    # 起点或终点缺失，跳过
                    pass
            else:
                valid_notes.append(note)
        
        return valid_notes
    
    def _check_playability(self, notes: List[Note]) -> List[Note]:
        """检查可玩性，移除不合理的配置"""
        # 按时间分组
        time_groups: Dict[float, List[Note]] = {}
        for note in notes:
            quantized_time = round(note.time * 48) / 48  # 量化到 1/48 拍
            if quantized_time not in time_groups:
                time_groups[quantized_time] = []
            time_groups[quantized_time].append(note)
        
        valid_notes = []
        
        for time, group in time_groups.items():
            # 检查同时音符数量（最多 2 只手）
            if len(group) > 4:
                # 保留前 4 个
                group = group[:4]
            
            # 检查位置冲突
            positions = set()
            filtered_group = []
            
            for note in group:
                if note.position not in positions:
                    positions.add(note.position)
                    filtered_group.append(note)
            
            valid_notes.extend(filtered_group)
        
        return valid_notes
    
    def _notes_to_simai(self, notes: List[Note], meta: ChartMeta) -> str:
        """将音符列表转换为完整的 simai 文本"""
        generator = SimaiGenerator(bpm=meta.bpm, divisor=4)
        return generator.generate(notes, meta)


class PostProcessor:
    """后处理器，用于优化生成的谱面"""
    
    @staticmethod
    def adjust_density(notes: List[Note], target_nps: float = 5.0) -> List[Note]:
        """
        调整谱面密度
        
        Args:
            notes: 音符列表
            target_nps: 目标每秒音符数
        
        Returns:
            调整后的音符列表
        """
        if not notes:
            return notes
        
        duration = notes[-1].time - notes[0].time
        current_nps = len(notes) / duration if duration > 0 else 0
        
        if current_nps <= target_nps:
            return notes
        
        # 需要减少音符
        keep_ratio = target_nps / current_nps
        
        # 保留 BREAK 和 SLIDE，随机删除普通 TAP
        important_notes = [n for n in notes if n.is_break or n.note_type == NoteType.SLIDE]
        other_notes = [n for n in notes if n not in important_notes]
        
        # 随机采样
        keep_count = int(len(other_notes) * keep_ratio)
        indices = np.random.choice(len(other_notes), keep_count, replace=False)
        kept_others = [other_notes[i] for i in sorted(indices)]
        
        result = important_notes + kept_others
        result.sort(key=lambda n: n.time)
        
        return result
    
    @staticmethod
    def add_breaks_at_climax(notes: List[Note], audio_features: AudioFeatures) -> List[Note]:
        """
        在高潮部分添加 BREAK
        
        Args:
            notes: 音符列表
            audio_features: 音频特征
        
        Returns:
            添加 BREAK 后的音符列表
        """
        # 找到能量峰值
        onset_strength = audio_features.onset_strength
        threshold = np.percentile(onset_strength, 90)
        
        # 找到高能量时间点
        high_energy_frames = np.where(onset_strength > threshold)[0]
        high_energy_times = set(
            round(audio_features.frame_times[f], 2)
            for f in high_energy_frames
        )
        
        # 标记这些位置的音符为 BREAK
        for note in notes:
            note_time = round(note.time, 2)
            if note_time in high_energy_times:
                note.is_break = True
        
        return notes


def generate_chart(
    audio_path: str,
    model_path: str,
    output_path: Optional[str] = None,
    config: Optional[GenerationConfig] = None
) -> str:
    """
    便捷函数：从音频生成谱面
    
    Args:
        audio_path: 音频文件路径
        model_path: 模型检查点路径
        output_path: 输出文件路径（可选）
        config: 生成配置
    
    Returns:
        生成的 simai 文本
    """
    generator = ChartGenerator(model_path)
    simai_text = generator.generate(audio_path, config)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(simai_text)
        print(f"谱面已保存到 {output_path}")
    
    return simai_text


def main():
    parser = argparse.ArgumentParser(description="生成 maimai 谱面")
    parser.add_argument("--audio", required=True, help="输入音频文件")
    parser.add_argument("--model", required=True, help="模型检查点路径")
    parser.add_argument("--output", help="输出文件路径")
    parser.add_argument("--title", help="曲目名称")
    parser.add_argument("--bpm", type=float, help="手动指定 BPM")
    parser.add_argument("--level", default="?", help="难度等级")
    parser.add_argument("--designer", default="AI", help="谱师名称")
    
    # 生成参数
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K 采样")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-P 采样")
    
    args = parser.parse_args()
    
    # 准备配置
    config = GenerationConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    
    # 准备元数据
    meta = ChartMeta(
        title=args.title or Path(args.audio).stem,
        bpm=args.bpm or 0,
        level=args.level,
        designer=args.designer
    )
    
    # 生成
    generator = ChartGenerator(args.model)
    simai_text = generator.generate(args.audio, config, meta)
    
    # 输出
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(simai_text)
        print(f"谱面已保存到 {args.output}")
    else:
        print(simai_text)


if __name__ == "__main__":
    main()
