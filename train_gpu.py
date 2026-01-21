#!/usr/bin/env python3
"""
GPU 训练启动脚本

优化的配置用于 GPU 训练，自动检测 GPU 并启用混合精度
"""

import torch
import argparse
import yaml
from pathlib import Path
from src.training.train import TrainingConfig, Trainer, main

def print_gpu_info():
    """打印 GPU 信息"""
    print("\n" + "="*60)
    print("GPU 信息")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA 可用")
        print(f"✓ GPU 数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  - 计算能力: {props.major}.{props.minor}")
            print(f"  - 总显存: {props.total_memory / 1024**3:.1f} GB")
            print(f"  - 当前显存占用: {torch.cuda.memory_allocated(i) / 1024**3:.1f} GB")
        
        print(f"\n✓ PyTorch 版本: {torch.__version__}")
        print(f"✓ 混合精度: 启用 (FP16)")
        
    else:
        print("✗ CUDA 不可用，将使用 CPU 训练")
        print(f"  PyTorch 版本: {torch.__version__}")
    
    print("="*60 + "\n")

def main_gpu():
    """主函数"""
    parser = argparse.ArgumentParser(description="GPU 训练")
    parser.add_argument("--config", type=str, help="配置文件路径 (yaml)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (默认: 8)")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率 (默认: 1e-4)")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数 (默认: 100)")
    parser.add_argument("--data-dir", default="data/processed", help="数据目录")
    parser.add_argument("--save-dir", default="models", help="模型保存目录")
    parser.add_argument("--no-mixed-precision", action="store_true", help="禁用混合精度")
    parser.add_argument("--use-wandb", action="store_true", help="使用 WandB 记录")
    
    args = parser.parse_args()
    
    # 如果指定了配置文件，则从配置文件加载参数
    if args.config and Path(args.config).exists():
        print(f"载入配置文件: {args.config}")
        with open(args.config, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            # 这里简化处理，只覆盖几个关键参数
            if 'training' in config_data:
                args.batch_size = config_data['training'].get('batch_size', args.batch_size)
                args.lr = config_data['training'].get('learning_rate', args.lr)
                args.epochs = config_data['training'].get('max_epochs', args.epochs)
            if 'data' in config_data:
                args.data_dir = config_data['data'].get('processed_dir', args.data_dir)

    # 打印 GPU 信息
    print_gpu_info()
    
    # 创建配置
    config = TrainingConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        save_dir=args.save_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        mixed_precision=torch.cuda.is_available() and not args.no_mixed_precision,
        use_wandb=args.use_wandb,
        gradient_accumulation_steps=4,  # 增加积累步数以节省显存
    )
    
    print("训练配置:")
    print(f"  - 设备: {config.device}")
    print(f"  - Batch Size: {config.batch_size}")
    print(f"  - 学习率: {config.learning_rate}")
    print(f"  - 混合精度: {'启用' if config.mixed_precision else '禁用'}")
    print(f"  - 最大轮数: {config.max_epochs}")
    print(f"  - 梯度累积步数: {config.gradient_accumulation_steps}")
    print()
    
    # 运行训练（注意：这里需要导入和创建模型及数据集）
    # 调用原始的 main 函数，但使用修改后的配置
    # 为了简化，我们直接运行训练脚本
    import sys
    sys.argv = [
        'train_gpu.py',
        '--batch-size', str(args.batch_size),
        '--learning-rate', str(args.lr),
        '--max-epochs', str(args.epochs),
        '--data-dir', args.data_dir,
        '--save-dir', args.save_dir,
    ]
    
    if args.use_wandb:
        sys.argv.append('--use-wandb')
    
    # 调用原始训练脚本的 main 函数
    main()

if __name__ == "__main__":
    main_gpu()

