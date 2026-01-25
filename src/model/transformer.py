"""
MaiChart AI 模型架构
基于 Transformer 的 Encoder-Decoder 架构，用于音频到谱面的生成
"""

import math
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat


@dataclass
class ModelConfig:
    """模型配置"""
    # 词表大小
    vocab_size: int = 256
    
    # 音频编码器
    audio_dim: int = 141  # n_mels(128) + onset(1) + chroma(12)
    audio_hidden_dim: int = 256
    audio_n_heads: int = 8
    audio_n_layers: int = 3
    audio_dropout: float = 0.1
    
    # 谱面解码器
    chart_hidden_dim: int = 256
    chart_n_heads: int = 8
    chart_n_layers: int = 4
    chart_dropout: float = 0.1
    
    # 位置编码
    max_audio_length: int = 12288  # 足够容纳最大 10296
    max_chart_length: int = 4096   # 实际最大 4096
    
    # 输出头
    position_classes: int = 33  # 8 + 25 touch positions
    action_classes: int = 20
    duration_classes: int = 10


class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 8192, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class AudioEncoder(nn.Module):
    """音频编码器"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 输入投影
        self.input_proj = nn.Linear(config.audio_dim, config.audio_hidden_dim)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(
            config.audio_hidden_dim,
            config.max_audio_length,
            config.audio_dropout
        )
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.audio_hidden_dim,
            nhead=config.audio_n_heads,
            dim_feedforward=config.audio_hidden_dim * 4,
            dropout=config.audio_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.audio_n_layers
        )
        
        # 输出层归一化
        self.norm = nn.LayerNorm(config.audio_hidden_dim)
    
    def forward(
        self,
        audio_features: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            audio_features: (batch, seq_len, audio_dim)
            audio_mask: (batch, seq_len) - 1 表示有效位置，0 表示填充
        
        Returns:
            encoded: (batch, seq_len, hidden_dim)
        """
        # 输入投影
        x = self.input_proj(audio_features)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # 构建注意力掩码
        if audio_mask is not None:
            # TransformerEncoder 期望的掩码格式：True 表示忽略的位置
            src_key_padding_mask = ~audio_mask.bool()
        else:
            src_key_padding_mask = None
        
        # 使用梯度检查点编码（节省显存）
        def encoder_fn(x, mask):
            return self.encoder(x, src_key_padding_mask=mask)
        
        if self.training:
            x = checkpoint(encoder_fn, x, src_key_padding_mask, use_reentrant=False)
        else:
            x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        return self.norm(x)


class ChartDecoder(nn.Module):
    """谱面解码器"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token 嵌入
        self.token_embedding = nn.Embedding(config.vocab_size, config.chart_hidden_dim)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(
            config.chart_hidden_dim,
            config.max_chart_length,
            config.chart_dropout
        )
        
        # 音频投影（如果维度不同）
        if config.audio_hidden_dim != config.chart_hidden_dim:
            self.audio_proj = nn.Linear(config.audio_hidden_dim, config.chart_hidden_dim)
        else:
            self.audio_proj = nn.Identity()
        
        # Transformer 解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.chart_hidden_dim,
            nhead=config.chart_n_heads,
            dim_feedforward=config.chart_hidden_dim * 4,
            dropout=config.chart_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.chart_n_layers
        )
        
        # 输出层归一化
        self.norm = nn.LayerNorm(config.chart_hidden_dim)
        
        # 输出头
        self.output_head = nn.Linear(config.chart_hidden_dim, config.vocab_size)
    
    def forward(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            tgt_tokens: (batch, tgt_len) - 目标 token 序列
            memory: (batch, src_len, hidden_dim) - 编码器输出
            tgt_mask: (batch, tgt_len) - 目标序列掩码
            memory_mask: (batch, src_len) - 编码器输出掩码
        
        Returns:
            logits: (batch, tgt_len, vocab_size)
        """
        # Token 嵌入
        x = self.token_embedding(tgt_tokens)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # 投影音频特征
        memory = self.audio_proj(memory)
        
        # 构建因果掩码
        tgt_len = tgt_tokens.size(1)
        causal_mask = self._generate_causal_mask(tgt_len, tgt_tokens.device)
        
        # 构建填充掩码
        if tgt_mask is not None:
            tgt_key_padding_mask = ~tgt_mask.bool()
        else:
            tgt_key_padding_mask = None
        
        if memory_mask is not None:
            memory_key_padding_mask = ~memory_mask.bool()
        else:
            memory_key_padding_mask = None
        
        # 使用梯度检查点解码（节省显存）
        def decoder_fn(x, mem, tgt_m, tgt_pm, mem_pm):
            return self.decoder(
                x, mem,
                tgt_mask=tgt_m,
                tgt_key_padding_mask=tgt_pm,
                memory_key_padding_mask=mem_pm
            )
        
        if self.training:
            x = checkpoint(decoder_fn, x, memory, causal_mask, tgt_key_padding_mask, 
                         memory_key_padding_mask, use_reentrant=False)
        else:
            x = self.decoder(
                x, memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
        
        x = self.norm(x)
        
        # 输出
        logits = self.output_head(x)
        
        return logits
    
    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """生成因果掩码"""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class MaiChartModel(nn.Module):
    """完整的谱面生成模型"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 编码器和解码器
        self.encoder = AudioEncoder(config)
        self.decoder = ChartDecoder(config)
    
    def forward(
        self,
        audio_features: torch.Tensor,
        chart_tokens: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        chart_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            audio_features: (batch, audio_len, audio_dim)
            chart_tokens: (batch, chart_len)
            audio_mask: (batch, audio_len)
            chart_mask: (batch, chart_len)
        
        Returns:
            包含 logits 和 loss 的字典
        """
        # 编码音频
        memory = self.encoder(audio_features, audio_mask)
        
        # 准备解码器输入（右移一位）
        decoder_input = chart_tokens[:, :-1]
        decoder_target = chart_tokens[:, 1:]
        
        if chart_mask is not None:
            decoder_mask = chart_mask[:, :-1]
            target_mask = chart_mask[:, 1:]
        else:
            decoder_mask = None
            target_mask = None
        
        # 解码
        logits = self.decoder(
            decoder_input,
            memory,
            tgt_mask=decoder_mask,
            memory_mask=audio_mask
        )
        
        # 计算损失
        loss = None
        if decoder_target is not None:
            # 解决 SEP token 过度主导损失的问题
            # 将 SEP (ID 3) 的权重调低，让模型更关注音符生成的准确性
            weights = torch.ones(self.config.vocab_size, device=logits.device)
            weights[3] = 0.1  # SEP 降权到 0.1
            
            # 你也可以选择给音符 token (如 TAP=100) 略微增权
            # weights[100:] *= 1.5 
            
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                decoder_target.reshape(-1),
                weight=weights,
                ignore_index=0  # 忽略 PAD
            )
        
        return {
            'logits': logits,
            'loss': loss,
            'memory': memory
        }
    
    @torch.no_grad()
    def generate(
        self,
        audio_features: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        max_length: int = 4096,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        """
        自回归生成谱面
        
        Args:
            audio_features: (batch, audio_len, audio_dim)
            audio_mask: (batch, audio_len)
            max_length: 最大生成长度
            temperature: 采样温度
            top_k: Top-K 采样
            top_p: Nucleus 采样
            bos_token_id: 开始 token ID
            eos_token_id: 结束 token ID
        
        Returns:
            generated: (batch, seq_len) - 生成的 token 序列
        """
        batch_size = audio_features.size(0)
        device = audio_features.device
        
        # 编码音频
        memory = self.encoder(audio_features, audio_mask)
        
        # 初始化序列
        generated = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=torch.long,
            device=device
        )
        
        # 跟踪哪些序列已完成
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length - 1):
            # 解码
            logits = self.decoder(generated, memory, memory_mask=audio_mask)
            
            # 获取最后一个位置的 logits
            next_logits = logits[:, -1, :] / temperature
            
            # Top-K 采样
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')
            
            # Top-P (Nucleus) 采样
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float('-inf')
            
            # 采样
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 对于已完成的序列，强制为 PAD
            next_token[finished] = 0
            
            # 更新完成状态
            finished = finished | (next_token.squeeze(-1) == eos_token_id)
            
            # 添加到序列
            generated = torch.cat([generated, next_token], dim=1)
            
            # 如果所有序列都完成，停止
            if finished.all():
                break
        
        return generated


def create_model(config: Optional[ModelConfig] = None) -> MaiChartModel:
    """创建模型"""
    if config is None:
        config = ModelConfig()
    return MaiChartModel(config)


if __name__ == "__main__":
    # 测试模型
    config = ModelConfig()
    model = create_model(config)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 测试前向传播
    batch_size = 2
    audio_len = 1000
    chart_len = 500
    
    audio_features = torch.randn(batch_size, audio_len, config.audio_dim)
    chart_tokens = torch.randint(0, config.vocab_size, (batch_size, chart_len))
    audio_mask = torch.ones(batch_size, audio_len)
    chart_mask = torch.ones(batch_size, chart_len)
    
    output = model(audio_features, chart_tokens, audio_mask, chart_mask)
    print(f"输出 logits 形状: {output['logits'].shape}")
    print(f"损失: {output['loss'].item():.4f}")
    
    # 测试生成
    generated = model.generate(
        audio_features[:1],
        audio_mask[:1],
        max_length=100,
        temperature=0.8
    )
    print(f"生成序列长度: {generated.shape[1]}")
