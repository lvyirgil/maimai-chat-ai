"""
音频特征提取模块
使用 librosa 提取用于谱面生成的音频特征
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError as e:
    LIBROSA_AVAILABLE = False
    warnings.warn(f"librosa 导入失败: {e}")

try:
    import madmom
    MADMOM_AVAILABLE = True
except (ImportError, Exception):
    MADMOM_AVAILABLE = False


@dataclass
class AudioFeatures:
    """音频特征容器"""
    # 基本信息
    duration: float  # 时长（秒）
    sample_rate: int
    
    # 时间序列特征
    mel_spectrogram: np.ndarray  # (n_mels, n_frames)
    onset_strength: np.ndarray   # (n_frames,)
    beat_times: np.ndarray       # (n_beats,)
    beat_frames: np.ndarray      # (n_beats,)
    tempo: float                 # BPM
    
    # 音高特征
    chroma: np.ndarray           # (12, n_frames)
    
    # 节奏特征
    tempogram: np.ndarray        # (win_length, n_frames)
    
    # 段落特征
    segment_boundaries: np.ndarray  # (n_segments,)
    segment_labels: np.ndarray      # (n_segments,)
    
    # 帧时间映射
    frame_times: np.ndarray      # (n_frames,)
    
    # 原始音频
    y: Optional[np.ndarray] = None
    
    def get_features_at_time(self, time: float) -> Dict[str, np.ndarray]:
        """获取指定时间点的特征"""
        frame_idx = np.argmin(np.abs(self.frame_times - time))
        return {
            'mel': self.mel_spectrogram[:, frame_idx],
            'onset': self.onset_strength[frame_idx],
            'chroma': self.chroma[:, frame_idx],
        }
    
    def get_features_in_range(self, start: float, end: float) -> Dict[str, np.ndarray]:
        """获取时间范围内的特征"""
        start_idx = np.argmin(np.abs(self.frame_times - start))
        end_idx = np.argmin(np.abs(self.frame_times - end))
        return {
            'mel': self.mel_spectrogram[:, start_idx:end_idx],
            'onset': self.onset_strength[start_idx:end_idx],
            'chroma': self.chroma[:, start_idx:end_idx],
            'times': self.frame_times[start_idx:end_idx],
        }


class AudioFeatureExtractor:
    """音频特征提取器"""
    
    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 512,
        n_mels: int = 128,
        n_fft: int = 2048,
        fmin: float = 20.0,
        fmax: float = 8000.0,
    ):
        if not LIBROSA_AVAILABLE:
            raise ImportError("请先安装 librosa: pip install librosa")
        
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.fmin = fmin
        self.fmax = fmax
        
    def extract(self, audio_path: str, keep_audio: bool = False) -> AudioFeatures:
        """从音频文件提取所有特征"""
        # 加载音频
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        duration = len(y) / sr
        
        # 计算帧时间
        n_frames = 1 + len(y) // self.hop_length
        frame_times = librosa.frames_to_time(
            np.arange(n_frames), sr=sr, hop_length=self.hop_length
        )
        
        # Mel 频谱图
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Onset 强度
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=self.hop_length
        )
        
        # 节拍检测
        tempo, beat_frames = librosa.beat.beat_track(
            y=y, sr=sr, hop_length=self.hop_length
        )
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=self.hop_length)
        
        # 色度特征
        chroma = librosa.feature.chroma_cqt(
            y=y, sr=sr, hop_length=self.hop_length
        )
        
        # Tempogram
        tempogram = librosa.feature.tempogram(
            y=y, sr=sr, hop_length=self.hop_length
        )
        
        # 段落分割
        segment_boundaries, segment_labels = self._detect_segments(y, sr)
        
        return AudioFeatures(
            duration=duration,
            sample_rate=sr,
            mel_spectrogram=mel_spec_db,
            onset_strength=onset_env,
            beat_times=beat_times,
            beat_frames=beat_frames,
            tempo=float(tempo) if isinstance(tempo, np.ndarray) else tempo,
            chroma=chroma,
            tempogram=tempogram,
            segment_boundaries=segment_boundaries,
            segment_labels=segment_labels,
            frame_times=frame_times,
            y=y if keep_audio else None
        )
    
    def _detect_segments(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """检测音乐段落"""
        # 使用频谱聚类进行段落分割
        # 计算 CQT
        C = np.abs(librosa.cqt(y=y, sr=sr, hop_length=self.hop_length))
        
        # 计算特征矩阵
        bounds = librosa.segment.agglomerative(C, k=8)
        bound_times = librosa.frames_to_time(bounds, sr=sr, hop_length=self.hop_length)
        
        # 简单的标签分配（基于 MFCC 聚类）
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=self.hop_length)
        
        # 为每个段落分配标签
        labels = np.zeros(len(bounds), dtype=int)
        for i, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
            segment_features = mfcc[:, start:end].mean(axis=1)
            # 简单分类：基于能量
            energy = np.mean(segment_features ** 2)
            labels[i] = int(energy > np.median(mfcc ** 2))
        
        return bound_times, labels
    
    def detect_onsets(self, audio_path: str) -> np.ndarray:
        """检测音频中的 onset 时间点"""
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr, hop_length=self.hop_length, backtrack=True
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)
        return onset_times
    
    def detect_beats(self, audio_path: str) -> Tuple[float, np.ndarray]:
        """检测节拍和 BPM"""
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=self.hop_length)
        return float(tempo) if isinstance(tempo, np.ndarray) else tempo, beat_times


class OnsetDetector:
    """高级 Onset 检测器，用于打击点候选生成"""
    
    def __init__(self, sample_rate: int = 22050, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
    
    def detect_all_onsets(self, audio_path: str) -> Dict[str, np.ndarray]:
        """检测不同类型的 onset"""
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # 分离谐波和打击成分
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # 打击 onset
        onset_percussive = librosa.onset.onset_detect(
            y=y_percussive, sr=sr, hop_length=self.hop_length,
            units='time'
        )
        
        # 谐波 onset（旋律变化）
        onset_harmonic = librosa.onset.onset_detect(
            y=y_harmonic, sr=sr, hop_length=self.hop_length,
            units='time'
        )
        
        # 综合 onset
        onset_all = librosa.onset.onset_detect(
            y=y, sr=sr, hop_length=self.hop_length,
            units='time'
        )
        
        return {
            'all': onset_all,
            'percussive': onset_percussive,
            'harmonic': onset_harmonic
        }
    
    def detect_with_strength(self, audio_path: str, threshold: float = 0.5) -> List[Tuple[float, float]]:
        """检测 onset 并返回强度"""
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # 计算 onset 强度
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        
        # 归一化
        onset_env = onset_env / onset_env.max() if onset_env.max() > 0 else onset_env
        
        # 峰值检测
        peaks = librosa.util.peak_pick(
            onset_env,
            pre_max=3, post_max=3,
            pre_avg=3, post_avg=5,
            delta=threshold, wait=10
        )
        
        # 转换为时间和强度
        times = librosa.frames_to_time(peaks, sr=sr, hop_length=self.hop_length)
        strengths = onset_env[peaks]
        
        return list(zip(times, strengths))


class SegmentAnalyzer:
    """音乐段落分析器"""
    
    SEGMENT_TYPES = ['intro', 'verse', 'chorus', 'bridge', 'outro', 'instrumental']
    
    def __init__(self, sample_rate: int = 22050, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
    
    def analyze(self, audio_path: str) -> List[Dict[str, Any]]:
        """分析音乐段落"""
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # 提取特征
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=self.hop_length)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
        
        # 自相似矩阵
        S = librosa.feature.stack_memory(mfcc, n_steps=3)
        R = librosa.segment.recurrence_matrix(S, mode='affinity', self=True)
        
        # 段落边界检测
        bounds = librosa.segment.agglomerative(mfcc, k=8)
        bound_times = librosa.frames_to_time(bounds, sr=sr, hop_length=self.hop_length)
        
        # 分析每个段落
        segments = []
        for i in range(len(bounds) - 1):
            start_frame = bounds[i]
            end_frame = bounds[i + 1]
            
            # 计算段落特征
            segment_mfcc = mfcc[:, start_frame:end_frame]
            segment_chroma = chroma[:, start_frame:end_frame]
            
            # 能量
            energy = np.mean(segment_mfcc[0] ** 2)
            
            # 色度变化（和弦丰富度）
            chroma_var = np.std(segment_chroma, axis=1).mean()
            
            # 简单分类
            if energy < 0.3:
                segment_type = 'intro' if i == 0 else 'bridge'
            elif energy > 0.7:
                segment_type = 'chorus'
            else:
                segment_type = 'verse'
            
            segments.append({
                'start': float(bound_times[i]),
                'end': float(bound_times[i + 1]),
                'type': segment_type,
                'energy': float(energy),
                'chroma_complexity': float(chroma_var),
            })
        
        return segments


def extract_features(audio_path: str) -> AudioFeatures:
    """便捷函数：提取音频特征"""
    extractor = AudioFeatureExtractor()
    return extractor.extract(audio_path)


def detect_onsets(audio_path: str) -> np.ndarray:
    """便捷函数：检测 onset"""
    extractor = AudioFeatureExtractor()
    return extractor.detect_onsets(audio_path)


def detect_beats(audio_path: str) -> Tuple[float, np.ndarray]:
    """便捷函数：检测节拍"""
    extractor = AudioFeatureExtractor()
    return extractor.detect_beats(audio_path)


if __name__ == "__main__":
    # 测试
    import sys
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        print(f"分析音频: {audio_path}")
        
        features = extract_features(audio_path)
        print(f"时长: {features.duration:.2f}s")
        print(f"BPM: {features.tempo:.1f}")
        print(f"节拍数: {len(features.beat_times)}")
        print(f"段落数: {len(features.segment_boundaries)}")
        
        # 检测 onset
        detector = OnsetDetector()
        onsets = detector.detect_with_strength(audio_path)
        print(f"检测到 {len(onsets)} 个打击点")
        
        # 段落分析
        analyzer = SegmentAnalyzer()
        segments = analyzer.analyze(audio_path)
        print("\n段落分析:")
        for seg in segments:
            print(f"  {seg['start']:.1f}s - {seg['end']:.1f}s: {seg['type']} (energy={seg['energy']:.2f})")
    else:
        print("使用方法: python audio_features.py <audio_file>")
