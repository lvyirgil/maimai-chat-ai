"""
Simai 专用 Tokenizer
将谱面数据转换为模型可处理的 token 序列
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np

from ..parser.simai_parser import Note, NoteType, Chart


class TokenType(Enum):
    """Token 类型"""
    # 特殊 token
    PAD = 0
    BOS = 1      # 序列开始
    EOS = 2      # 序列结束
    SEP = 3      # 分隔符（时间推进）
    EACH = 4     # 多押分隔符
    
    # 位置 token
    POS_1 = 10
    POS_2 = 11
    POS_3 = 12
    POS_4 = 13
    POS_5 = 14
    POS_6 = 15
    POS_7 = 16
    POS_8 = 17
    
    # Touch 位置
    TOUCH_A1 = 20
    TOUCH_A2 = 21
    TOUCH_A3 = 22
    TOUCH_A4 = 23
    TOUCH_A5 = 24
    TOUCH_A6 = 25
    TOUCH_A7 = 26
    TOUCH_A8 = 27
    TOUCH_B1 = 30
    TOUCH_B2 = 31
    TOUCH_B3 = 32
    TOUCH_B4 = 33
    TOUCH_B5 = 34
    TOUCH_B6 = 35
    TOUCH_B7 = 36
    TOUCH_B8 = 37
    TOUCH_C = 40
    TOUCH_D1 = 50
    TOUCH_D2 = 51
    TOUCH_D3 = 52
    TOUCH_D4 = 53
    TOUCH_D5 = 54
    TOUCH_D6 = 55
    TOUCH_D7 = 56
    TOUCH_D8 = 57
    TOUCH_E1 = 60
    TOUCH_E2 = 61
    TOUCH_E3 = 62
    TOUCH_E4 = 63
    TOUCH_E5 = 64
    TOUCH_E6 = 65
    TOUCH_E7 = 66
    TOUCH_E8 = 67
    
    # 动作 token
    TAP = 100
    HOLD_START = 101
    HOLD_END = 102
    SLIDE_START = 103
    SLIDE_END = 104
    TOUCH = 105
    TOUCH_HOLD_START = 106
    TOUCH_HOLD_END = 107
    
    # 修饰符
    BREAK = 110
    EX = 111
    STAR = 112
    STAR_ROTATE = 113
    FIREWORK = 114
    
    # Slide 路径
    SLIDE_STRAIGHT = 120    # -
    SLIDE_ARC_AUTO = 121    # ^
    SLIDE_ARC_CCW = 122     # <
    SLIDE_ARC_CW = 123      # >
    SLIDE_V_CENTER = 124    # v
    SLIDE_P = 125           # p
    SLIDE_Q = 126           # q
    SLIDE_S = 127           # s
    SLIDE_Z = 128           # z
    SLIDE_PP = 129          # pp
    SLIDE_QQ = 130          # qq
    SLIDE_V_CHAIN = 131     # V
    SLIDE_WIFI = 132        # w
    
    # 时值 token (以 1/192 为单位的离散值)
    DUR_1 = 200      # 全音符 (192 ticks)
    DUR_2 = 201      # 二分音符 (96 ticks)
    DUR_4 = 202      # 四分音符 (48 ticks)
    DUR_8 = 203      # 八分音符 (24 ticks)
    DUR_16 = 204     # 十六分音符 (12 ticks)
    DUR_32 = 205     # 三十二分音符 (6 ticks)
    DUR_3 = 206      # 三连音
    DUR_6 = 207      # 六连音
    DUR_12 = 208     # 十二连音
    DUR_24 = 209     # 二十四连音


@dataclass
class Token:
    """单个 token"""
    type: TokenType
    value: int  # token ID
    raw: str = ""  # 原始文本
    
    def __repr__(self):
        return f"Token({self.type.name}, {self.value})"


@dataclass
class TokenizedChart:
    """Tokenize 后的谱面"""
    tokens: List[Token]
    # 元数据
    bpm: float
    duration: float
    # 映射
    token_ids: np.ndarray
    attention_mask: np.ndarray


class SimaiTokenizer:
    """Simai 谱面 Tokenizer"""
    
    # 时间网格：1/192 分音符
    TIME_GRID = 192
    
    # 位置映射
    POSITION_TO_TOKEN = {
        '1': TokenType.POS_1, '2': TokenType.POS_2, '3': TokenType.POS_3, '4': TokenType.POS_4,
        '5': TokenType.POS_5, '6': TokenType.POS_6, '7': TokenType.POS_7, '8': TokenType.POS_8,
    }
    
    TOUCH_TO_TOKEN = {
        'A1': TokenType.TOUCH_A1, 'A2': TokenType.TOUCH_A2, 'A3': TokenType.TOUCH_A3, 'A4': TokenType.TOUCH_A4,
        'A5': TokenType.TOUCH_A5, 'A6': TokenType.TOUCH_A6, 'A7': TokenType.TOUCH_A7, 'A8': TokenType.TOUCH_A8,
        'B1': TokenType.TOUCH_B1, 'B2': TokenType.TOUCH_B2, 'B3': TokenType.TOUCH_B3, 'B4': TokenType.TOUCH_B4,
        'B5': TokenType.TOUCH_B5, 'B6': TokenType.TOUCH_B6, 'B7': TokenType.TOUCH_B7, 'B8': TokenType.TOUCH_B8,
        'C': TokenType.TOUCH_C,
        'D1': TokenType.TOUCH_D1, 'D2': TokenType.TOUCH_D2, 'D3': TokenType.TOUCH_D3, 'D4': TokenType.TOUCH_D4,
        'D5': TokenType.TOUCH_D5, 'D6': TokenType.TOUCH_D6, 'D7': TokenType.TOUCH_D7, 'D8': TokenType.TOUCH_D8,
        'E1': TokenType.TOUCH_E1, 'E2': TokenType.TOUCH_E2, 'E3': TokenType.TOUCH_E3, 'E4': TokenType.TOUCH_E4,
        'E5': TokenType.TOUCH_E5, 'E6': TokenType.TOUCH_E6, 'E7': TokenType.TOUCH_E7, 'E8': TokenType.TOUCH_E8,
    }
    
    SLIDE_PATH_TO_TOKEN = {
        '-': TokenType.SLIDE_STRAIGHT,
        '^': TokenType.SLIDE_ARC_AUTO,
        '<': TokenType.SLIDE_ARC_CCW,
        '>': TokenType.SLIDE_ARC_CW,
        'v': TokenType.SLIDE_V_CENTER,
        'p': TokenType.SLIDE_P,
        'q': TokenType.SLIDE_Q,
        's': TokenType.SLIDE_S,
        'z': TokenType.SLIDE_Z,
        'pp': TokenType.SLIDE_PP,
        'qq': TokenType.SLIDE_QQ,
        'V': TokenType.SLIDE_V_CHAIN,
        'w': TokenType.SLIDE_WIFI,
    }
    
    # 时值映射（ticks -> TokenType）
    DURATION_TO_TOKEN = {
        192: TokenType.DUR_1,
        96: TokenType.DUR_2,
        64: TokenType.DUR_3,   # 三连音
        48: TokenType.DUR_4,
        32: TokenType.DUR_6,   # 六连音
        24: TokenType.DUR_8,
        16: TokenType.DUR_12,  # 十二连音
        12: TokenType.DUR_16,
        8: TokenType.DUR_24,   # 二十四连音
        6: TokenType.DUR_32,
    }
    
    def __init__(self, max_seq_length: int = 4096):
        self.max_seq_length = max_seq_length
        
        # 构建词表
        self.vocab = self._build_vocab()
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
    
    def _build_vocab(self) -> Dict[TokenType, int]:
        """构建词表"""
        vocab = {}
        for token_type in TokenType:
            vocab[token_type] = token_type.value
        return vocab
    
    def tokenize(self, chart: Chart) -> TokenizedChart:
        """将谱面转换为 token 序列"""
        tokens = []
        
        # 添加 BOS
        tokens.append(Token(TokenType.BOS, self.vocab[TokenType.BOS]))
        
        # 按时间排序音符
        sorted_notes = sorted(chart.notes, key=lambda n: n.time)
        
        # 量化时间网格（基于 BPM）
        beat_duration = 60.0 / chart.meta.bpm  # 一拍的时长
        tick_duration = beat_duration / 48  # 一个 1/48 拍的时长
        
        # 将音符按时间分组
        time_groups: Dict[int, List[Note]] = {}
        for note in sorted_notes:
            # 量化到最近的 1/48 拍
            tick = int(round(note.time / tick_duration))
            if tick not in time_groups:
                time_groups[tick] = []
            time_groups[tick].append(note)
        
        # 按时间顺序处理
        sorted_ticks = sorted(time_groups.keys())
        last_tick = 0
        
        for tick in sorted_ticks:
            # 添加时间分隔符
            time_diff = tick - last_tick
            while time_diff > 0:
                tokens.append(Token(TokenType.SEP, self.vocab[TokenType.SEP]))
                time_diff -= 1
            
            # 处理该时间点的所有音符
            notes_at_tick = time_groups[tick]
            for i, note in enumerate(notes_at_tick):
                if i > 0:
                    # 多押分隔符
                    tokens.append(Token(TokenType.EACH, self.vocab[TokenType.EACH]))
                
                # 转换音符
                note_tokens = self._note_to_tokens(note)
                tokens.extend(note_tokens)
            
            last_tick = tick
        
        # 添加 EOS
        tokens.append(Token(TokenType.EOS, self.vocab[TokenType.EOS]))
        
        # 转换为数组
        token_ids = np.array([t.value for t in tokens], dtype=np.int32)
        attention_mask = np.ones(len(tokens), dtype=np.int32)
        
        # Padding
        if len(token_ids) < self.max_seq_length:
            pad_length = self.max_seq_length - len(token_ids)
            token_ids = np.pad(token_ids, (0, pad_length), constant_values=self.vocab[TokenType.PAD])
            attention_mask = np.pad(attention_mask, (0, pad_length), constant_values=0)
        elif len(token_ids) > self.max_seq_length:
            token_ids = token_ids[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
        
        return TokenizedChart(
            tokens=tokens,
            bpm=chart.meta.bpm,
            duration=chart.notes[-1].time if chart.notes else 0,
            token_ids=token_ids,
            attention_mask=attention_mask
        )
    
    def _note_to_tokens(self, note: Note) -> List[Token]:
        """将单个音符转换为 token 序列"""
        tokens = []
        
        # 修饰符
        if note.is_break:
            tokens.append(Token(TokenType.BREAK, self.vocab[TokenType.BREAK]))
        if note.is_ex:
            tokens.append(Token(TokenType.EX, self.vocab[TokenType.EX]))
        
        # 位置
        if note.position in self.POSITION_TO_TOKEN:
            pos_token = self.POSITION_TO_TOKEN[note.position]
            tokens.append(Token(pos_token, self.vocab[pos_token]))
        elif note.position in self.TOUCH_TO_TOKEN:
            pos_token = self.TOUCH_TO_TOKEN[note.position]
            tokens.append(Token(pos_token, self.vocab[pos_token]))
        
        # 动作类型
        if note.note_type == NoteType.TAP:
            tokens.append(Token(TokenType.TAP, self.vocab[TokenType.TAP]))
            if note.is_star:
                star_token = TokenType.STAR_ROTATE if note.star_rotating else TokenType.STAR
                tokens.append(Token(star_token, self.vocab[star_token]))
        
        elif note.note_type == NoteType.HOLD:
            tokens.append(Token(TokenType.HOLD_START, self.vocab[TokenType.HOLD_START]))
            if note.hold_duration and note.hold_duration.ticks > 0:
                dur_token = self._ticks_to_duration_token(note.hold_duration.ticks)
                if dur_token:
                    tokens.append(Token(dur_token, self.vocab[dur_token]))
        
        elif note.note_type == NoteType.SLIDE:
            tokens.append(Token(TokenType.SLIDE_START, self.vocab[TokenType.SLIDE_START]))
            
            # 解析 slide 路径
            if note.slide_path:
                for path_char in ['-', '^', '<', '>', 'v', 'V', 'pp', 'qq', 'p', 'q', 's', 'z', 'w']:
                    if path_char in note.slide_path:
                        if path_char in self.SLIDE_PATH_TO_TOKEN:
                            path_token = self.SLIDE_PATH_TO_TOKEN[path_char]
                            tokens.append(Token(path_token, self.vocab[path_token]))
                            break
            
            # 终点
            if note.slide_end and note.slide_end in self.POSITION_TO_TOKEN:
                end_token = self.POSITION_TO_TOKEN[note.slide_end]
                tokens.append(Token(end_token, self.vocab[end_token]))
            
            # 时长
            if note.slide_duration and note.slide_duration.ticks > 0:
                dur_token = self._ticks_to_duration_token(note.slide_duration.ticks)
                if dur_token:
                    tokens.append(Token(dur_token, self.vocab[dur_token]))
        
        elif note.note_type == NoteType.TOUCH:
            tokens.append(Token(TokenType.TOUCH, self.vocab[TokenType.TOUCH]))
            if note.is_firework:
                tokens.append(Token(TokenType.FIREWORK, self.vocab[TokenType.FIREWORK]))
        
        elif note.note_type == NoteType.TOUCH_HOLD:
            tokens.append(Token(TokenType.TOUCH_HOLD_START, self.vocab[TokenType.TOUCH_HOLD_START]))
            if note.hold_duration and note.hold_duration.ticks > 0:
                dur_token = self._ticks_to_duration_token(note.hold_duration.ticks)
                if dur_token:
                    tokens.append(Token(dur_token, self.vocab[dur_token]))
        
        return tokens
    
    def _ticks_to_duration_token(self, ticks: int) -> Optional[TokenType]:
        """将 ticks 转换为最接近的时值 token"""
        # 找到最接近的时值
        min_diff = float('inf')
        best_token = None
        
        for dur_ticks, token in self.DURATION_TO_TOKEN.items():
            diff = abs(ticks - dur_ticks)
            if diff < min_diff:
                min_diff = diff
                best_token = token
        
        return best_token
    
    def detokenize(self, token_ids: np.ndarray, bpm: float = 120.0) -> List[Note]:
        """将 token 序列转换回音符列表"""
        notes = []
        current_time = 0.0
        beat_duration = 60.0 / bpm
        tick_duration = beat_duration / 48
        
        i = 0
        while i < len(token_ids):
            token_id = int(token_ids[i])
            
            # 跳过 padding 和特殊 token
            if token_id == self.vocab[TokenType.PAD]:
                break
            if token_id == self.vocab[TokenType.BOS]:
                i += 1
                continue
            if token_id == self.vocab[TokenType.EOS]:
                break
            if token_id == self.vocab[TokenType.SEP]:
                current_time += tick_duration
                i += 1
                continue
            if token_id == self.vocab[TokenType.EACH]:
                i += 1
                continue
            
            # 解析音符
            note, consumed = self._parse_note_tokens(token_ids[i:], current_time)
            if note:
                notes.append(note)
            i += max(consumed, 1)
        
        return notes
    
    def _parse_note_tokens(self, token_ids: np.ndarray, time: float) -> Tuple[Optional[Note], int]:
        """从 token 序列解析单个音符"""
        if len(token_ids) == 0:
            return None, 0
        
        consumed = 0
        is_break = False
        is_ex = False
        position = None
        note_type = None
        is_star = False
        star_rotating = False
        hold_duration = None
        slide_path = None
        slide_end = None
        slide_duration = None
        is_firework = False
        
        while consumed < len(token_ids):
            token_id = int(token_ids[consumed])
            
            # 检查是否是分隔符
            if token_id in [self.vocab[TokenType.SEP], self.vocab[TokenType.EACH], 
                           self.vocab[TokenType.EOS], self.vocab[TokenType.PAD]]:
                break
            
            # 修饰符
            if token_id == self.vocab[TokenType.BREAK]:
                is_break = True
            elif token_id == self.vocab[TokenType.EX]:
                is_ex = True
            elif token_id == self.vocab[TokenType.STAR]:
                is_star = True
            elif token_id == self.vocab[TokenType.STAR_ROTATE]:
                is_star = True
                star_rotating = True
            elif token_id == self.vocab[TokenType.FIREWORK]:
                is_firework = True
            
            # 位置
            elif token_id in [self.vocab[t] for t in self.POSITION_TO_TOKEN.values()]:
                for pos, tok in self.POSITION_TO_TOKEN.items():
                    if token_id == self.vocab[tok]:
                        if position is None:
                            position = pos
                        else:
                            slide_end = pos
                        break
            elif token_id in [self.vocab[t] for t in self.TOUCH_TO_TOKEN.values()]:
                for pos, tok in self.TOUCH_TO_TOKEN.items():
                    if token_id == self.vocab[tok]:
                        position = pos
                        break
            
            # 动作类型
            elif token_id == self.vocab[TokenType.TAP]:
                note_type = NoteType.TAP
            elif token_id == self.vocab[TokenType.HOLD_START]:
                note_type = NoteType.HOLD
            elif token_id == self.vocab[TokenType.SLIDE_START]:
                note_type = NoteType.SLIDE
            elif token_id == self.vocab[TokenType.TOUCH]:
                note_type = NoteType.TOUCH
            elif token_id == self.vocab[TokenType.TOUCH_HOLD_START]:
                note_type = NoteType.TOUCH_HOLD
            
            # Slide 路径
            elif token_id in [self.vocab[t] for t in self.SLIDE_PATH_TO_TOKEN.values()]:
                for path, tok in self.SLIDE_PATH_TO_TOKEN.items():
                    if token_id == self.vocab[tok]:
                        slide_path = path
                        break
            
            consumed += 1
        
        if position and note_type:
            from ..parser.simai_parser import Duration
            note = Note(
                time=time,
                position=position,
                note_type=note_type,
                is_break=is_break,
                is_ex=is_ex,
                hold_duration=hold_duration,
                slide_path=slide_path,
                slide_end=slide_end,
                slide_duration=slide_duration,
                is_star=is_star,
                star_rotating=star_rotating,
                is_firework=is_firework
            )
            return note, consumed
        
        return None, consumed
    
    def encode(self, text: str) -> np.ndarray:
        """编码 simai 文本"""
        from ..parser.simai_parser import parse_simai
        chart = parse_simai(text)
        tokenized = self.tokenize(chart)
        return tokenized.token_ids
    
    def decode(self, token_ids: np.ndarray, bpm: float = 120.0) -> str:
        """解码为 simai 文本"""
        from ..parser.simai_parser import generate_simai, ChartMeta
        notes = self.detokenize(token_ids, bpm)
        meta = ChartMeta(bpm=bpm)
        return generate_simai(notes, meta, bpm)


# 便捷函数
def create_tokenizer(max_seq_length: int = 4096) -> SimaiTokenizer:
    """创建 tokenizer"""
    return SimaiTokenizer(max_seq_length)


if __name__ == "__main__":
    # 测试
    tokenizer = SimaiTokenizer()
    print(f"词表大小: {tokenizer.vocab_size}")
    
    # 测试编码
    test_chart = """
&title=Test
&wholebpm=150
&first=0.5
&lv_5=12+
&des_5=AI
&inote_5=
(150){4}
1,2,3,4,
1/5,,
1h[4:2],,
1-5[4:1],
E
"""
    
    token_ids = tokenizer.encode(test_chart)
    print(f"Token 数量: {np.count_nonzero(token_ids)}")
    
    # 测试解码
    decoded = tokenizer.decode(token_ids, bpm=150)
    print(f"解码结果:\n{decoded[:500]}...")
