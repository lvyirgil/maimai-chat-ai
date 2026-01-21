@echo off
REM 官方谱面导入快速启动脚本
REM 此脚本会自动导入 D:\BaiduNetdiskDownload\官谱 中的所有谱面和音频

setlocal enabledelayedexpansion

echo ==========================================
echo   MaiChart AI - 官方谱面导入工具
echo ==========================================
echo.

REM 检查是否在项目目录
if not exist "data\raw" (
    echo 错误: 未找到 data\raw 目录
    echo 请确保在项目根目录运行此脚本
    pause
    exit /b 1
)

REM 检查压缩包目录
if not exist "D:\BaiduNetdiskDownload\官谱" (
    echo 警告: 未找到压缩包目录 D:\BaiduNetdiskDownload\官谱
    echo.
    echo 请先下载官方谱面压缩包到该目录
    pause
    exit /b 1
)

echo 即将导入官方谱面...
echo 这可能需要 30 分钟到 1 小时（取决于磁盘和网络速度）
echo.
pause

REM 运行导入脚本
python scripts/import_official_charts.py %*

if %errorlevel% equ 0 (
    echo.
    echo ==========================================
    echo   导入完成！
    echo ==========================================
    echo.
    echo 下一步：
    echo 1. 运行数据预处理: python -m src.data.preprocess
    echo 2. 开始训练: python -m src.training.train
    echo.
) else (
    echo.
    echo 导入过程中出现错误，请查看上面的信息
    echo.
)

pause
