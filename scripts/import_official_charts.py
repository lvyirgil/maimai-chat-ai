#!/usr/bin/env python3
"""
官方谱面导入脚本

将压缩包中的 simai 谱面和音频文件提取到项目的数据文件夹中。
压缩包结构:
  版本名/
    ├── 歌曲名/
    │   ├── maidata.txt (simai 谱面)
    │   └── track.mp3 (音频)
    └── ...
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path
from typing import List, Tuple
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChartImporter:
    """官方谱面导入器"""
    
    def __init__(self, 
                 source_dir: str = r"D:\BaiduNetdiskDownload\官谱",
                 raw_dir: str = "data/raw",
                 audio_dir: str = "data/audio"):
        """
        初始化导入器
        
        Args:
            source_dir: 源压缩包目录
            raw_dir: 谱面输出目录（相对于项目根目录）
            audio_dir: 音频输出目录（相对于项目根目录）
        """
        self.source_dir = Path(source_dir)
        self.raw_dir = Path(raw_dir).absolute()
        self.audio_dir = Path(audio_dir).absolute()
        
        # 创建目录如果不存在
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"源目录: {self.source_dir}")
        logger.info(f"谱面输出目录: {self.raw_dir}")
        logger.info(f"音频输出目录: {self.audio_dir}")
    
    def get_zip_files(self) -> List[Path]:
        """获取所有 zip 文件"""
        if not self.source_dir.exists():
            logger.error(f"源目录不存在: {self.source_dir}")
            return []
        
        zip_files = list(self.source_dir.glob("*.zip"))
        logger.info(f"找到 {len(zip_files)} 个压缩包")
        return sorted(zip_files)
    
    def extract_and_import(self, zip_path: Path, skip_existing: bool = True) -> Tuple[int, int]:
        """
        从 zip 文件提取并导入谱面和音频
        
        Args:
            zip_path: zip 文件路径
            skip_existing: 是否跳过已存在的文件
            
        Returns:
            (导入成功数, 跳过数)
        """
        logger.info(f"\n处理: {zip_path.name}")
        success_count = 0
        skip_count = 0
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # 获取版本名称（如 "01. maimai"）
                namelist = zip_ref.namelist()
                version_folders = set()
                
                for name in namelist:
                    parts = Path(name).parts
                    if len(parts) > 0:
                        version_folders.add(parts[0])
                
                logger.info(f"发现 {len(version_folders)} 个版本")
                
                # 遍历每个版本的歌曲
                for version_folder in version_folders:
                    for name in namelist:
                        path = Path(name)
                        
                        # 匹配: version/song_name/maidata.txt 或 version/song_name/track.mp3
                        if len(path.parts) == 3 and path.parts[0] == version_folder:
                            song_name = path.parts[1]
                            filename = path.name
                            
                            if filename == "maidata.txt":
                                # 处理谱面文件
                                target_path = self.raw_dir / f"{song_name}.txt"
                                
                                if target_path.exists() and skip_existing:
                                    logger.debug(f"跳过已存在的谱面: {song_name}")
                                    skip_count += 1
                                else:
                                    try:
                                        content = zip_ref.read(name)
                                        target_path.write_bytes(content)
                                        logger.info(f"✓ 导入谱面: {song_name}")
                                        success_count += 1
                                    except Exception as e:
                                        logger.error(f"✗ 导入谱面失败 {song_name}: {e}")
                            
                            elif filename == "track.mp3":
                                # 处理音频文件
                                target_path = self.audio_dir / f"{song_name}.mp3"
                                
                                if target_path.exists() and skip_existing:
                                    logger.debug(f"跳过已存在的音频: {song_name}")
                                    skip_count += 1
                                else:
                                    try:
                                        content = zip_ref.read(name)
                                        target_path.write_bytes(content)
                                        logger.info(f"✓ 导入音频: {song_name}")
                                        success_count += 1
                                    except Exception as e:
                                        logger.error(f"✗ 导入音频失败 {song_name}: {e}")
        
        except zipfile.BadZipFile:
            logger.error(f"✗ 无效的 zip 文件: {zip_path}")
        except Exception as e:
            logger.error(f"✗ 处理失败: {e}")
        
        return success_count, skip_count
    
    def import_all(self, skip_existing: bool = True) -> None:
        """导入所有压缩包"""
        zip_files = self.get_zip_files()
        
        if not zip_files:
            logger.warning("未找到 zip 文件")
            return
        
        total_success = 0
        total_skip = 0
        
        for zip_path in zip_files:
            success, skip = self.extract_and_import(zip_path, skip_existing)
            total_success += success
            total_skip += skip
        
        logger.info(f"\n{'='*50}")
        logger.info(f"导入完成!")
        logger.info(f"成功导入: {total_success} 个文件")
        logger.info(f"跳过已存在: {total_skip} 个文件")
        logger.info(f"谱面目录: {self.raw_dir}")
        logger.info(f"音频目录: {self.audio_dir}")
        
        # 统计导入结果
        chart_count = len(list(self.raw_dir.glob("*.txt")))
        audio_count = len(list(self.audio_dir.glob("*.mp3")))
        logger.info(f"当前谱面数: {chart_count}")
        logger.info(f"当前音频数: {audio_count}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="导入官方谱面")
    parser.add_argument(
        "--source",
        default=r"D:\BaiduNetdiskDownload\官谱",
        help="压缩包源目录"
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="谱面输出目录"
    )
    parser.add_argument(
        "--audio-dir",
        default="data/audio",
        help="音频输出目录"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的文件"
    )
    
    args = parser.parse_args()
    
    importer = ChartImporter(
        source_dir=args.source,
        raw_dir=args.raw_dir,
        audio_dir=args.audio_dir
    )
    
    importer.import_all(skip_existing=not args.overwrite)


if __name__ == "__main__":
    main()
