#!/usr/bin/env python3
"""检查 CUDA 可用性和 PyTorch 配置"""

import torch
import sys

print("=" * 60)
print("PyTorch CUDA 诊断")
print("=" * 60)

print(f"\n1. PyTorch 版本: {torch.__version__}")
print(f"2. CUDA 可用: {torch.cuda.is_available()}")
print(f"3. CUDA 设备数: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"4. 当前 CUDA 设备: {torch.cuda.current_device()}")
    print(f"5. CUDA 设备名称: {torch.cuda.get_device_name(0)}")
    print(f"6. CUDA 计算能力: {torch.cuda.get_device_capability(0)}")
    
    # 获取 CUDA 内存信息
    print(f"\n内存信息:")
    print(f"   - 总内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"   - 可用内存: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB (已分配)")
    print(f"   - 保留内存: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB (已保留)")
else:
    print("\n⚠️  CUDA 不可用!")
    print("\n解决方案:")
    print("1. 检查 NVIDIA 驱动: nvidia-smi")
    print("2. 重新安装 PyTorch with CUDA support:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("3. 检查 NVIDIA 卡是否被识别: nvidia-smi")

print("\n" + "=" * 60)
