#!/usr/bin/env python3
"""GPU 训练启动脚本"""

import torch
import subprocess
import sys

def check_gpu():
    """检查 GPU 可用性"""
    print("\n" + "="*70)
    print("GPU 环境检查")
    print("="*70)
    
    if torch.cuda.is_available():
        print("✓ CUDA 可用")
        print(f"✓ GPU 数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\n  GPU {i}: {props.name}")
            print(f"    - 计算能力: {props.major}.{props.minor}")
            print(f"    - 总显存: {props.total_memory / 1024**3:.1f} GB")
        
        print(f"\n✓ PyTorch 版本: {torch.__version__}")
        print(f"✓ CUDA 版本: {torch.version.cuda}")
        print(f"✓ cuDNN 版本: {torch.backends.cudnn.version()}")
        print(f"\n✓ 混合精度 (AMP): 启用")
        print(f"✓ 优化 Batch Size: 16 (GPU)")
        print(f"✓ 减少梯度累积: 2 步 (GPU)")
        
    else:
        print("⚠ CUDA 不可用，将使用 CPU 训练（速度较慢）")
        print(f"  PyTorch 版本: {torch.__version__}")
    
    print("="*70 + "\n")

def main():
    check_gpu()
    
    # 默认使用 GPU 优化参数
    cmd = [
        sys.executable, '-m', 'src.training.train',
        '--batch-size', '16',  # GPU 默认 batch size
        '--hidden-dim', '256',  # GPU 优化参数
        '--n-layers', '3',     # GPU 优化参数
    ]
    
    # 添加自定义参数
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    
    print("启动训练命令:")
    print(f"  {' '.join(cmd)}")
    print()
    
    # 运行训练
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
