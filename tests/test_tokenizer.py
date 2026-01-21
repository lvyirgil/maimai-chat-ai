"""
单元测试 - Tokenizer
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '..')

from src.data import SimaiTokenizer, TokenType
from src.parser import parse_simai


class TestTokenizer:
    """测试 Tokenizer"""
    
    def setup_method(self):
        """每个测试前初始化"""
        self.tokenizer = SimaiTokenizer(max_seq_length=1024)
    
    def test_vocab_size(self):
        """测试词表大小"""
        assert self.tokenizer.vocab_size > 0
        assert len(self.tokenizer.vocab) == self.tokenizer.vocab_size
    
    def test_tokenize_simple(self):
        """测试简单谱面的 tokenize"""
        chart_text = """
&title=Test
&wholebpm=120
&first=0
&inote_5=
(120){4}
1,2,3,4,
E
"""
        chart = parse_simai(chart_text)
        tokenized = self.tokenizer.tokenize(chart)
        
        # 检查输出形状
        assert tokenized.token_ids.shape == (1024,)
        assert tokenized.attention_mask.shape == (1024,)
        
        # 检查特殊 token
        assert tokenized.token_ids[0] == self.tokenizer.vocab[TokenType.BOS]
    
    def test_tokenize_with_each(self):
        """测试带多押的 tokenize"""
        chart_text = """
&title=Test
&wholebpm=120
&first=0
&inote_5=
(120){4}
1/5,
E
"""
        chart = parse_simai(chart_text)
        tokenized = self.tokenizer.tokenize(chart)
        
        # 应该包含 EACH token
        token_types = [t.type for t in tokenized.tokens]
        assert TokenType.EACH in token_types
    
    def test_detokenize(self):
        """测试 detokenize"""
        chart_text = """
&title=Test
&wholebpm=120
&first=0
&inote_5=
(120){4}
1,2,3,
E
"""
        chart = parse_simai(chart_text)
        tokenized = self.tokenizer.tokenize(chart)
        
        # Detokenize
        notes = self.tokenizer.detokenize(tokenized.token_ids, bpm=120)
        
        # 应该有音符
        assert len(notes) > 0
    
    def test_encode_decode(self):
        """测试编码解码"""
        chart_text = """
&title=Test
&wholebpm=120
&first=0
&inote_5=
(120){4}
1,2,
E
"""
        # 编码
        token_ids = self.tokenizer.encode(chart_text)
        
        assert isinstance(token_ids, np.ndarray)
        assert token_ids.shape == (self.tokenizer.max_seq_length,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
