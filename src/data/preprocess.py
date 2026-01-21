"""
数据预处理主入口
"""

import argparse
from pathlib import Path

from .dataset import ChartDataset, DatasetConfig


def main():
    parser = argparse.ArgumentParser(description="预处理谱面数据")
    parser.add_argument("--raw-dir", default="data/raw", help="原始谱面目录")
    parser.add_argument("--audio-dir", default="data/audio", help="音频文件目录")
    parser.add_argument("--output-dir", default="data/processed", help="输出目录")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="最大序列长度")
    
    args = parser.parse_args()
    
    config = DatasetConfig(
        raw_dir=args.raw_dir,
        audio_dir=args.audio_dir,
        processed_dir=args.output_dir,
        max_seq_length=args.max_seq_length
    )
    
    dataset = ChartDataset(config)
    dataset.process_all(save=True)
    
    print("数据预处理完成!")


if __name__ == "__main__":
    main()
