"""
单元测试 - 模型
"""

import pytest
import torch
import sys
sys.path.insert(0, '..')

from src.model import MaiChartModel, ModelConfig, create_model


class TestModel:
    """测试模型"""
    
    def setup_method(self):
        """每个测试前初始化"""
        self.config = ModelConfig(
            vocab_size=256,
            audio_dim=141,
            audio_hidden_dim=128,
            chart_hidden_dim=128,
            audio_n_heads=4,
            chart_n_heads=4,
            audio_n_layers=2,
            chart_n_layers=2
        )
        self.model = create_model(self.config)
    
    def test_model_creation(self):
        """测试模型创建"""
        assert isinstance(self.model, MaiChartModel)
        
        # 检查参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        assert total_params > 0
    
    def test_forward_pass(self):
        """测试前向传播"""
        batch_size = 2
        audio_len = 100
        chart_len = 50
        
        audio_features = torch.randn(batch_size, audio_len, self.config.audio_dim)
        chart_tokens = torch.randint(0, self.config.vocab_size, (batch_size, chart_len))
        audio_mask = torch.ones(batch_size, audio_len)
        chart_mask = torch.ones(batch_size, chart_len)
        
        output = self.model(audio_features, chart_tokens, audio_mask, chart_mask)
        
        # 检查输出
        assert 'logits' in output
        assert 'loss' in output
        assert output['logits'].shape == (batch_size, chart_len - 1, self.config.vocab_size)
    
    def test_generate(self):
        """测试生成"""
        batch_size = 1
        audio_len = 50
        
        audio_features = torch.randn(batch_size, audio_len, self.config.audio_dim)
        audio_mask = torch.ones(batch_size, audio_len)
        
        generated = self.model.generate(
            audio_features,
            audio_mask,
            max_length=20,
            temperature=1.0
        )
        
        # 检查输出形状
        assert generated.shape[0] == batch_size
        assert generated.shape[1] <= 20
        
        # 检查开始 token
        assert generated[0, 0].item() == 1  # BOS
    
    def test_gradient_flow(self):
        """测试梯度流动"""
        batch_size = 2
        audio_len = 50
        chart_len = 30
        
        audio_features = torch.randn(batch_size, audio_len, self.config.audio_dim)
        chart_tokens = torch.randint(0, self.config.vocab_size, (batch_size, chart_len))
        
        output = self.model(audio_features, chart_tokens)
        loss = output['loss']
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        has_grad = False
        for param in self.model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad, "模型应该有梯度"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
