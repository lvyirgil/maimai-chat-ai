@echo off
REM GPU 训练启动脚本（Windows 批处理）

echo.
echo ======================================================================
echo   MaiChart AI - GPU 训练启动
echo ======================================================================
echo.

REM 检查 Python 和 PyTorch
python -c "import torch; print('✓ CUDA 可用:', torch.cuda.is_available()); print('✓ GPU 数量:', torch.cuda.device_count()); print('✓ PyTorch:', torch.__version__)" 2>nul

if errorlevel 1 (
    echo 错误: 无法检测 CUDA 环境
    pause
    exit /b 1
)

echo.
echo 使用以下参数启动 GPU 训练:
echo   - Batch Size: 16 (GPU 优化)
echo   - 隐藏维度: 256 (GPU 优化)
echo   - Transformer 层数: 3 (GPU 优化)
echo   - 混合精度: 启用
echo.

REM 运行训练
python -m src.training.train ^
    --batch-size 16 ^
    --hidden-dim 256 ^
    --n-layers 3 ^
    %*

if errorlevel 1 (
    echo.
    echo 训练过程中出现错误，请查看上面的信息
)

pause
