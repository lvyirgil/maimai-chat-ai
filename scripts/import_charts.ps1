#!/usr/bin/env powershell
<#
.SYNOPSIS
    官方谱面导入工具 - PowerShell 版本

.DESCRIPTION
    快速将官方谱面压缩包导入到项目数据目录

.EXAMPLE
    .\import_charts.ps1

.EXAMPLE
    .\import_charts.ps1 -Source "D:\my_charts" -Overwrite

.NOTES
    运行此脚本需要管理员权限（如果涉及权限相关操作）
#>

param(
    [string]$Source = "D:\BaiduNetdiskDownload\官谱",
    [string]$RawDir = "data/raw",
    [string]$AudioDir = "data/audio",
    [switch]$Overwrite
)

# 设置脚本停止行为
$ErrorActionPreference = "Continue"

# 输出标题
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  MaiChart AI - 官方谱面导入工具" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查项目目录
if (-not (Test-Path "data/raw")) {
    Write-Host "错误: 未找到 data/raw 目录" -ForegroundColor Red
    Write-Host "请确保在项目根目录运行此脚本" -ForegroundColor Red
    exit 1
}

# 检查源目录
if (-not (Test-Path $Source)) {
    Write-Host "警告: 未找到压缩包目录 $Source" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "请先下载官方谱面压缩包到该目录" -ForegroundColor Yellow
    exit 1
}

# 构建命令
$pythonArgs = @("scripts/import_official_charts.py")
$pythonArgs += "--source", "`"$Source`""
$pythonArgs += "--raw-dir", "`"$RawDir`""
$pythonArgs += "--audio-dir", "`"$AudioDir`""

if ($Overwrite) {
    $pythonArgs += "--overwrite"
}

Write-Host "源目录: $Source" -ForegroundColor Green
Write-Host "谱面输出: $RawDir" -ForegroundColor Green
Write-Host "音频输出: $AudioDir" -ForegroundColor Green
Write-Host ""

if ($Overwrite) {
    Write-Host "模式: 覆盖已存在的文件" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "这可能需要 30 分钟到 1 小时（取决于磁盘和网络速度）" -ForegroundColor Cyan
Write-Host ""

# 确认
$confirm = Read-Host "继续? (Y/N)"
if ($confirm -ne "Y" -and $confirm -ne "y") {
    Write-Host "已取消" -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "开始导入..." -ForegroundColor Green
Write-Host ""

# 运行导入脚本
& python @pythonArgs

# 检查返回值
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  导入完成！" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "下一步:" -ForegroundColor Cyan
    Write-Host "1. 运行数据预处理: python -m src.data.preprocess" -ForegroundColor White
    Write-Host "2. 开始训练: python -m src.training.train" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "导入过程中出现错误，请查看上面的信息" -ForegroundColor Red
    Write-Host ""
    exit 1
}
