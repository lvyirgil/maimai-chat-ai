"""
MajdataView 集成模块
支持自动预览和导出
"""

import os
import json
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class MajdataConfig:
    """Majdata 配置"""
    # Majdata 安装路径
    majdata_path: str = ""
    # MajdataView 可执行文件
    view_executable: str = "MajdataView.exe"
    # 谱面存放目录
    charts_dir: str = "charts"


class MajdataExporter:
    """Majdata 格式导出器"""
    
    def __init__(self, config: Optional[MajdataConfig] = None):
        self.config = config or MajdataConfig()
    
    def export(
        self,
        simai_text: str,
        audio_path: str,
        output_dir: str,
        song_name: Optional[str] = None,
        cover_path: Optional[str] = None,
        bg_path: Optional[str] = None
    ) -> str:
        """
        导出为 Majdata 格式
        
        Args:
            simai_text: simai 谱面文本
            audio_path: 音频文件路径
            output_dir: 输出目录
            song_name: 歌曲名称
            cover_path: 封面图片路径
            bg_path: 背景图片路径
        
        Returns:
            输出目录路径
        """
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 确定歌曲名称
        if song_name is None:
            song_name = Path(audio_path).stem
        
        # 创建歌曲目录
        song_dir = output_path / song_name
        song_dir.mkdir(exist_ok=True)
        
        # 复制音频文件
        audio_ext = Path(audio_path).suffix
        audio_dest = song_dir / f"track{audio_ext}"
        shutil.copy2(audio_path, audio_dest)
        
        # 保存谱面文件
        chart_path = song_dir / "maidata.txt"
        with open(chart_path, 'w', encoding='utf-8') as f:
            f.write(simai_text)
        
        # 复制封面（如果有）
        if cover_path and Path(cover_path).exists():
            cover_ext = Path(cover_path).suffix
            cover_dest = song_dir / f"bg{cover_ext}"
            shutil.copy2(cover_path, cover_dest)
        
        # 复制背景（如果有）
        if bg_path and Path(bg_path).exists():
            bg_ext = Path(bg_path).suffix
            bg_dest = song_dir / f"mv{bg_ext}"
            shutil.copy2(bg_path, bg_dest)
        
        print(f"已导出到 {song_dir}")
        return str(song_dir)
    
    def launch_preview(self, chart_dir: str) -> bool:
        """
        启动 MajdataView 预览
        
        Args:
            chart_dir: 谱面目录
        
        Returns:
            是否成功启动
        """
        if not self.config.majdata_path:
            print("未配置 Majdata 路径")
            return False
        
        view_path = Path(self.config.majdata_path) / self.config.view_executable
        
        if not view_path.exists():
            print(f"找不到 MajdataView: {view_path}")
            return False
        
        try:
            # 启动 MajdataView
            subprocess.Popen([str(view_path), chart_dir])
            print(f"已启动 MajdataView 预览: {chart_dir}")
            return True
        except Exception as e:
            print(f"启动失败: {e}")
            return False


class AstroDXExporter:
    """AstroDX 格式导出器"""
    
    def export(
        self,
        simai_text: str,
        audio_path: str,
        output_dir: str,
        song_name: Optional[str] = None
    ) -> str:
        """
        导出为 AstroDX 格式
        
        AstroDX 使用与 Majdata 相同的格式结构
        
        Args:
            simai_text: simai 谱面文本
            audio_path: 音频文件路径
            output_dir: 输出目录
            song_name: 歌曲名称
        
        Returns:
            输出目录路径
        """
        # AstroDX 格式与 Majdata 基本相同
        exporter = MajdataExporter()
        return exporter.export(simai_text, audio_path, output_dir, song_name)
    
    def create_song_info(
        self,
        output_dir: str,
        title: str,
        artist: str = "",
        bpm: float = 120,
        **kwargs
    ) -> None:
        """
        创建 AstroDX 歌曲信息文件
        
        Args:
            output_dir: 输出目录
            title: 歌曲标题
            artist: 艺术家
            bpm: BPM
        """
        info = {
            "title": title,
            "artist": artist,
            "bpm": bpm,
            **kwargs
        }
        
        info_path = Path(output_dir) / "songinfo.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)


def export_for_majdata(
    simai_text: str,
    audio_path: str,
    output_dir: str,
    **kwargs
) -> str:
    """便捷函数：导出为 Majdata 格式"""
    exporter = MajdataExporter()
    return exporter.export(simai_text, audio_path, output_dir, **kwargs)


def export_for_astrodx(
    simai_text: str,
    audio_path: str,
    output_dir: str,
    **kwargs
) -> str:
    """便捷函数：导出为 AstroDX 格式"""
    exporter = AstroDXExporter()
    return exporter.export(simai_text, audio_path, output_dir, **kwargs)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="导出谱面到 Majdata/AstroDX 格式")
    parser.add_argument("--chart", required=True, help="simai 谱面文件")
    parser.add_argument("--audio", required=True, help="音频文件")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--name", help="歌曲名称")
    parser.add_argument("--format", choices=['majdata', 'astrodx'], default='majdata',
                       help="输出格式")
    parser.add_argument("--preview", action="store_true", help="导出后启动预览")
    parser.add_argument("--majdata-path", help="Majdata 安装路径")
    
    args = parser.parse_args()
    
    # 读取谱面
    with open(args.chart, 'r', encoding='utf-8') as f:
        simai_text = f.read()
    
    # 导出
    if args.format == 'majdata':
        config = MajdataConfig(majdata_path=args.majdata_path or "")
        exporter = MajdataExporter(config)
        output = exporter.export(simai_text, args.audio, args.output, args.name)
        
        if args.preview:
            exporter.launch_preview(output)
    else:
        exporter = AstroDXExporter()
        output = exporter.export(simai_text, args.audio, args.output, args.name)
    
    print(f"完成! 输出目录: {output}")
