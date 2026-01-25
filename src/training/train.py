"""
模型训练脚本
"""

import os
import time
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from ..model.transformer import MaiChartModel, ModelConfig, create_model
from ..data.dataset import ChartDataset, DatasetConfig, create_dataset


@dataclass
class TrainingConfig:
    """训练配置"""
    # 数据
    data_dir: str = "data/processed"
    
    # 训练参数
    batch_size: int = 16  # GPU 训练，增加 batch size
    learning_rate: float = 5e-5  # 从 1e-4 降低到 5e-5 以适应权重调整后的微调阶段
    warmup_steps: int = 1000
    max_epochs: int = 200  # 增加最大轮数以支持更长时间的微调
    max_grad_norm: float = 1.0
    
    # 优化器
    weight_decay: float = 0.01
    
    # 保存
    save_dir: str = "models"
    save_every: int = 5  # 每 N 个 epoch 保存一次
    
    # 日志
    log_every: int = 100
    use_wandb: bool = False
    wandb_project: str = "maichart-ai"
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True  # 启用混合精度加速 GPU 训练
    
    # 早停
    early_stopping_patience: int = 10


class Trainer:
    """模型训练器"""
    
    def __init__(
        self,
        model: MaiChartModel,
        train_dataset: ChartDataset,
        val_dataset: Optional[ChartDataset],
        config: TrainingConfig
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        # 移动模型到设备
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)
        
        # 混合精度训练：仅在 CUDA 上启用
        self.use_mixed_precision = config.mixed_precision and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_mixed_precision)
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # 检测是否在 Colab 环境
        self.in_colab = 'google.colab' in sys.modules
        
        # 保存目录优先级：网盘 > 本地
        if self.in_colab:
            drive_path = Path("/content/drive/MyDrive/maimai/models")
            if Path("/content/drive").exists():
                self.save_dir = drive_path
                print(f"检测到网盘已挂载，保存路径设为: {self.save_dir}")
            else:
                self.save_dir = Path(config.save_dir)
                print(f"网盘未挂载，将保存到本地并在每轮后触发下载: {self.save_dir}")
        else:
            self.save_dir = Path(config.save_dir)
            
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # WandB
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=config.wandb_project,
                config={**asdict(config), **asdict(model.config)}
            )
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            return 1.0
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # 创建批次生成器
        batch_generator = self.train_dataset.batch_generator(
            self.config.batch_size,
            shuffle=True
        )
        
        progress_bar = tqdm(
            batch_generator,
            desc=f"Epoch {self.epoch}",
            total=len(self.train_dataset) // self.config.batch_size
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            # 移动数据到设备
            audio_features = torch.tensor(batch['audio_features'], dtype=torch.float32).to(self.device)
            audio_mask = torch.tensor(batch['audio_mask'], dtype=torch.float32).to(self.device)
            chart_tokens = torch.tensor(batch['chart_tokens'], dtype=torch.long).to(self.device)
            chart_mask = torch.tensor(batch['attention_mask'], dtype=torch.float32).to(self.device)
            
            # 混合精度前向传播
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_mixed_precision):
                output = self.model(
                    audio_features=audio_features,
                    chart_tokens=chart_tokens,
                    audio_mask=audio_mask,
                    chart_mask=chart_mask
                )
                loss = output['loss'] / self.config.gradient_accumulation_steps
            
            # 反向传播
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # 梯度裁剪
                if self.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                # 更新参数
                if self.use_mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # 记录
            total_loss += output['loss'].item()
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{output['loss'].item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # 日志
            if self.global_step % self.config.log_every == 0:
                if self.config.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        'train/loss': output['loss'].item(),
                        'train/lr': self.scheduler.get_last_lr()[0],
                        'train/step': self.global_step
                    })
        
        return total_loss / max(num_batches, 1)
    
    @torch.no_grad()
    @torch.no_grad()
    def validate(self) -> float:
        """验证"""
        if self.val_dataset is None or len(self.val_dataset) == 0:
            return 0.0
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        batch_generator = self.val_dataset.batch_generator(
            self.config.batch_size,
            shuffle=False
        )
        
        for batch in tqdm(batch_generator, desc="Validating"):
            audio_features = torch.tensor(batch['audio_features'], dtype=torch.float32).to(self.device)
            audio_mask = torch.tensor(batch['audio_mask'], dtype=torch.float32).to(self.device)
            chart_tokens = torch.tensor(batch['chart_tokens'], dtype=torch.long).to(self.device)
            chart_mask = torch.tensor(batch['attention_mask'], dtype=torch.float32).to(self.device)
            
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_mixed_precision):
                output = self.model(
                    audio_features=audio_features,
                    chart_tokens=chart_tokens,
                    audio_mask=audio_mask,
                    chart_mask=chart_mask
                )
            
            total_loss += output['loss'].item()
            num_batches += 1
            
            # 清空显存缓存
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return total_loss / max(num_batches, 1)
    
    def save_checkpoint(self, name: str = "checkpoint"):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'model_config': asdict(self.model.config),
            'training_config': asdict(self.config)
        }
        
        path = self.save_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        print(f"检查点已保存到 {path}")
        
        # Colab 环境触发下载
        if self.in_colab:
            try:
                from google.colab import files
                files.download(str(path))
                print(f"已触发下载: {path.name}")
            except Exception as e:
                # 某些环境下导出可能失败，静默处理或仅打印
                pass
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"从 {path} 加载检查点 (epoch {self.epoch})")
    
    def train(self):
        """完整训练流程"""
        print(f"开始训练，设备: {self.device}")
        print(f"训练集大小: {len(self.train_dataset)}")
        if self.val_dataset:
            print(f"验证集大小: {len(self.val_dataset)}")
        
        # 从加载的 epoch 开始，如果是新训练则从 0 开始
        start_epoch = self.epoch + 1 if self.global_step > 0 else 0
        
        for epoch in range(start_epoch, self.config.max_epochs):
            self.epoch = epoch
            
            # 训练
            train_loss = self.train_epoch()
            print(f"Epoch {epoch}: train_loss = {train_loss:.4f}")
            
            # 验证
            val_loss = self.validate()
            if val_loss > 0:
                print(f"Epoch {epoch}: val_loss = {val_loss:.4f}")
            
            # 日志
            if self.config.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'epoch': epoch,
                    'train/epoch_loss': train_loss,
                    'val/loss': val_loss
                })
            
            # 保存最佳模型 (包含优化器状态)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # 每一轮训练完都输出 epoch 检查点
            self.save_checkpoint(f"epoch_{epoch}")
            
            # 早停
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"早停于 epoch {epoch}")
                break
        
        # 保存最终模型
        self.save_checkpoint("final")
        
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="训练 MaiChart 模型")
    
    # 数据参数
    parser.add_argument("--data-dir", default="data/processed", help="处理后的数据目录")
    
    # 训练参数
    parser.add_argument("--batch-size", type=int, default=16, help="批大小 (GPU 默认 16，CPU 默认 4)")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--max-epochs", type=int, default=100, help="最大训练轮数")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="预热步数")
    
    # 模型参数
    parser.add_argument("--hidden-dim", type=int, default=256, help="隐藏层维度 (GPU 优化为 256)")
    parser.add_argument("--n-heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--n-layers", type=int, default=3, help="层数 (GPU 优化为 3)")
    
    # 保存参数
    parser.add_argument("--save-dir", default="models", help="模型保存目录")
    parser.add_argument("--resume", type=str, help="从检查点恢复训练")
    
    # 日志参数
    parser.add_argument("--use-wandb", action="store_true", help="使用 WandB 记录")
    parser.add_argument("--no-mixed-precision", action="store_true", help="禁用混合精度 (GPU)")
    
    args = parser.parse_args()
    
    # 自动检测设备和调整参数
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # GPU 自动调整参数
    if device == "cuda":
        if args.batch_size == 16:  # 使用默认值
            args.batch_size = 16
        print(f"✓ 检测到 GPU，启用混合精度和优化参数")
    else:
        if args.batch_size == 16:  # 使用默认值
            args.batch_size = 4
        print(f"✓ 使用 CPU 训练，使用较小的 batch size")
    
    # 创建配置
    training_config = TrainingConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        warmup_steps=args.warmup_steps,
        save_dir=args.save_dir,
        use_wandb=args.use_wandb,
        device=device,
        mixed_precision=device == "cuda" and not args.no_mixed_precision,
        gradient_accumulation_steps=2 if device == "cuda" else 8  # GPU 减少累积
    )
    
    model_config = ModelConfig(
        audio_hidden_dim=args.hidden_dim,
        chart_hidden_dim=args.hidden_dim,
        audio_n_heads=args.n_heads,
        chart_n_heads=args.n_heads,
        audio_n_layers=args.n_layers,
        chart_n_layers=args.n_layers
    )
    
    # 加载数据
    dataset_config = DatasetConfig(processed_dir=args.data_dir)
    dataset = create_dataset(dataset_config)
    
    # 尝试加载已处理的数据
    try:
        dataset.load()
    except:
        print("未找到预处理数据，请先运行 python -m src.data.preprocess")
        return
    
    # 划分数据集
    train_dataset, val_dataset, test_dataset = dataset.split()
    
    # 创建模型
    model = create_model(model_config)
    
    # 创建训练器
    trainer = Trainer(model, train_dataset, val_dataset, training_config)
    
    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
