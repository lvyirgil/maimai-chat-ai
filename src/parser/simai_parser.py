"""
Simai 格式谱面解析器
支持完整的 simai 语法，包括 TAP, HOLD, SLIDE, TOUCH 等所有音符类型
"""

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Dict, Any
import json


class NoteType(Enum):
    """音符类型枚举"""
    TAP = auto()
    HOLD = auto()
    SLIDE = auto()
    TOUCH = auto()
    TOUCH_HOLD = auto()
    BREAK = auto()
    EX = auto()
    STAR = auto()  # 星星（带旋转或不带）
    

class SlideType(Enum):
    """Slide 路径类型"""
    STRAIGHT = "-"       # 直线
    ARC_AUTO = "^"       # 自动选择最近弧线
    ARC_CCW = "<"        # 逆时针弧线
    ARC_CW = ">"         # 顺时针弧线
    V_CENTER = "v"       # 通过中心折线
    P_CURVE = "p"        # B区逆时针弧线
    Q_CURVE = "q"        # B区顺时针弧线
    S_ZIGZAG = "s"       # S型折线
    Z_ZIGZAG = "z"       # Z型折线
    PP_CURVE = "pp"      # 大逆时针弧线
    QQ_CURVE = "qq"      # 大顺时针弧线
    V_CHAIN = "V"        # 链式折线 (xVyz)
    WIFI = "w"           # 扇形星星


@dataclass
class Duration:
    """时值表示"""
    # 使用 1/192 作为最小单位
    ticks: int = 0
    # 或者直接秒数
    seconds: Optional[float] = None
    # 原始表示
    raw: str = ""
    
    @classmethod
    def from_beat(cls, divisor: int, count: int = 1, bpm: float = 120) -> 'Duration':
        """从拍数创建时值，如 [4:1] 表示 1 个四分音符"""
        ticks = int(192 / divisor * count)
        seconds = 60 / bpm / divisor * count
        return cls(ticks=ticks, seconds=seconds, raw=f"[{divisor}:{count}]")
    
    @classmethod
    def from_seconds(cls, seconds: float) -> 'Duration':
        """从秒数创建时值"""
        return cls(ticks=0, seconds=seconds, raw=f"[#{seconds}]")


@dataclass
class Note:
    """音符基类"""
    # 时间位置（秒）
    time: float
    # 键位 (1-8 for TAP/HOLD/SLIDE, A1-E8/C for TOUCH)
    position: str
    # 音符类型
    note_type: NoteType
    # 是否为 BREAK
    is_break: bool = False
    # 是否为 EX
    is_ex: bool = False
    # Hold 时长
    hold_duration: Optional[Duration] = None
    # Slide 信息
    slide_path: Optional[str] = None
    slide_end: Optional[str] = None
    slide_duration: Optional[Duration] = None
    slide_wait: Optional[Duration] = None
    # 特殊标记
    is_star: bool = False
    star_rotating: bool = False
    is_firework: bool = False  # 烟花 (f)
    # 伪多押标记
    is_fake_each: bool = False
    # 原始文本
    raw: str = ""


@dataclass 
class ChartMeta:
    """谱面元数据"""
    title: str = ""
    artist: str = ""
    bpm: float = 120.0
    offset: float = 0.0  # first 偏移
    level: str = ""
    designer: str = ""
    difficulty: int = 5  # 1-6 对应 Easy-ReMaster


@dataclass
class Chart:
    """完整谱面"""
    meta: ChartMeta
    notes: List[Note] = field(default_factory=list)
    # BPM 变化列表 [(time, bpm), ...]
    bpm_changes: List[Tuple[float, float]] = field(default_factory=list)
    

class SimaiParser:
    """Simai 格式解析器"""
    
    # 位置正则
    POSITION_PATTERN = re.compile(r'^[1-8]$')
    TOUCH_PATTERN = re.compile(r'^([A-E])([1-8])|C$')
    
    # 时值正则 [X:Y] 或 [#X:Y] 或 [#X] 或 [BPM#X:Y]
    DURATION_PATTERN = re.compile(
        r'\[(?:(\d+(?:\.\d+)?)\#)?(\d+)(?::(\d+))?\]|\[\#(\d+(?:\.\d+)?)(?::(\d+(?:\.\d+)?))?\]'
    )
    
    # Slide 路径正则
    SLIDE_PATH_PATTERN = re.compile(r'[-^<>vVpqszw]|pp|qq')
    
    def __init__(self):
        self.current_bpm = 120.0
        self.current_divisor = 4  # 当前分音符
        self.current_time = 0.0
        self.notes: List[Note] = []
        self.bpm_changes: List[Tuple[float, float]] = []
        
    def parse(self, simai_text: str) -> Chart:
        """解析完整的 simai 文本"""
        self.notes = []
        self.bpm_changes = []
        self.current_time = 0.0
        self.current_bpm = 120.0
        self.current_divisor = 4
        
        meta = ChartMeta()
        chart_data = ""
        
        # 解析文件头
        lines = simai_text.strip().split('\n')
        in_notes = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('&title='):
                meta.title = line[7:]
            elif line.startswith('&wholebpm='):
                meta.bpm = float(line[10:])
                self.current_bpm = meta.bpm
            elif line.startswith('&first='):
                try:
                    meta.offset = float(line[7:]) if line[7:] else 0.0
                except:
                    meta.offset = 0.0
            elif line.startswith('&lv_5='):
                meta.level = line[6:]
            elif line.startswith('&des_5='):
                meta.designer = line[7:]
            elif line.startswith('&inote_5='):
                in_notes = True
                chart_data = line[9:]
            elif in_notes:
                if line.startswith('&'):
                    break
                chart_data += line
        
        # 设置初始 BPM
        self.bpm_changes.append((0.0, self.current_bpm))
        self.current_time = meta.offset
        
        # 解析谱面数据
        self._parse_chart_data(chart_data)
        
        return Chart(meta=meta, notes=self.notes, bpm_changes=self.bpm_changes)
    
    def _parse_chart_data(self, data: str):
        """解析谱面主体数据"""
        # 移除空白字符（但保留逗号）
        data = re.sub(r'[\s\n]+', '', data)
        
        i = 0
        while i < len(data):
            char = data[i]
            
            # BPM 变化 (XXX)
            if char == '(':
                end = data.find(')', i)
                if end != -1:
                    bpm_str = data[i+1:end]
                    try:
                        self.current_bpm = float(bpm_str)
                        self.bpm_changes.append((self.current_time, self.current_bpm))
                    except ValueError:
                        pass
                    i = end + 1
                    continue
            
            # 分音符变化 {X} 或 {#X}
            elif char == '{':
                end = data.find('}', i)
                if end != -1:
                    divisor_str = data[i+1:end]
                    if divisor_str.startswith('#'):
                        # 绝对时间间隔
                        self.current_divisor = -float(divisor_str[1:])
                    else:
                        self.current_divisor = int(divisor_str)
                    i = end + 1
                    continue
            
            # 时间推进（逗号）
            elif char == ',':
                self._advance_time()
                i += 1
                continue
            
            # 音符或多押
            else:
                # 收集到下一个逗号或结束
                note_end = i
                bracket_depth = 0
                while note_end < len(data):
                    c = data[note_end]
                    if c == '[':
                        bracket_depth += 1
                    elif c == ']':
                        bracket_depth -= 1
                    elif c == ',' and bracket_depth == 0:
                        break
                    elif c in '({' and bracket_depth == 0:
                        break
                    note_end += 1
                
                note_str = data[i:note_end]
                if note_str:
                    self._parse_notes(note_str)
                i = note_end
                continue
    
    def _advance_time(self):
        """根据当前分音符推进时间"""
        if isinstance(self.current_divisor, float) and self.current_divisor < 0:
            # 绝对时间
            self.current_time += -self.current_divisor
        else:
            # 相对时间（基于 BPM 和分音符）
            beat_duration = 60.0 / self.current_bpm
            self.current_time += beat_duration * 4 / self.current_divisor
    
    def _parse_notes(self, note_str: str):
        """解析单个时间点的所有音符（可能包含多押）"""
        # 按 / 分割多押
        parts = note_str.split('/')
        
        for part in parts:
            if not part:
                continue
            self._parse_single_note(part.strip())
    
    def _parse_single_note(self, note_str: str):
        """解析单个音符"""
        if not note_str:
            return
            
        # 检查伪多押
        is_fake_each = '`' in note_str
        note_str = note_str.replace('`', '')
        
        # 检查前缀
        is_break = note_str.startswith('b')
        if is_break:
            note_str = note_str[1:]
        
        is_ex = note_str.startswith('x')
        if is_ex:
            note_str = note_str[1:]
        
        if not note_str:
            return
            
        # 判断音符类型
        first_char = note_str[0]
        
        # Touch 音符 (A-E 或 C)
        if first_char in 'ABCDE':
            self._parse_touch(note_str, is_break, is_ex, is_fake_each)
        # 普通位置音符 (1-8)
        elif first_char.isdigit():
            self._parse_position_note(note_str, is_break, is_ex, is_fake_each)
    
    def _parse_touch(self, note_str: str, is_break: bool, is_ex: bool, is_fake_each: bool):
        """解析 Touch 音符"""
        # 提取位置
        if note_str.startswith('C'):
            position = 'C'
            remaining = note_str[1:]
        else:
            match = re.match(r'^([A-E])([1-8])', note_str)
            if not match:
                return
            position = match.group(1) + match.group(2)
            remaining = note_str[len(position):]
        
        # 检查烟花
        is_firework = 'f' in remaining
        remaining = remaining.replace('f', '')
        
        # 检查 Hold
        if 'h' in remaining:
            note_type = NoteType.TOUCH_HOLD
            duration = self._parse_duration(remaining)
        else:
            note_type = NoteType.TOUCH
            duration = None
        
        note = Note(
            time=self.current_time,
            position=position,
            note_type=note_type,
            is_break=is_break,
            is_ex=is_ex,
            hold_duration=duration,
            is_firework=is_firework,
            is_fake_each=is_fake_each,
            raw=note_str
        )
        self.notes.append(note)
    
    def _parse_position_note(self, note_str: str, is_break: bool, is_ex: bool, is_fake_each: bool):
        """解析位置音符 (TAP/HOLD/SLIDE)"""
        position = note_str[0]
        remaining = note_str[1:]
        
        # 检查特殊标记
        is_star = False
        star_rotating = False
        star_hidden = False
        
        if '$$' in remaining:
            is_star = True
            star_rotating = True
            remaining = remaining.replace('$$', '')
        elif '$' in remaining:
            is_star = True
            remaining = remaining.replace('$', '')
        
        if '@' in remaining:
            # Star 转 Tap
            remaining = remaining.replace('@', '')
        if '?' in remaining:
            star_hidden = True
            remaining = remaining.replace('?', '')
        if '!' in remaining:
            remaining = remaining.replace('!', '')
        
        # 检查是否有 Slide
        slide_match = self.SLIDE_PATH_PATTERN.search(remaining)
        
        if slide_match:
            # Slide 音符
            self._parse_slide(position, remaining, is_break, is_ex, is_star, star_rotating, is_fake_each, note_str)
        elif 'h' in remaining:
            # Hold 音符
            duration = self._parse_duration(remaining)
            note = Note(
                time=self.current_time,
                position=position,
                note_type=NoteType.HOLD,
                is_break=is_break,
                is_ex=is_ex,
                hold_duration=duration,
                is_star=is_star,
                star_rotating=star_rotating,
                is_fake_each=is_fake_each,
                raw=note_str
            )
            self.notes.append(note)
        else:
            # Tap 音符
            note = Note(
                time=self.current_time,
                position=position,
                note_type=NoteType.TAP if not is_star else NoteType.STAR,
                is_break=is_break,
                is_ex=is_ex,
                is_star=is_star,
                star_rotating=star_rotating,
                is_fake_each=is_fake_each,
                raw=note_str
            )
            self.notes.append(note)
    
    def _parse_slide(self, start_pos: str, remaining: str, is_break: bool, is_ex: bool,
                     is_star: bool, star_rotating: bool, is_fake_each: bool, raw: str):
        """解析 Slide 音符"""
        # 解析所有 slide 路径（可能有多段）
        # 例如: 1-7<1qq6[4:1]
        
        # 简化处理：找到第一个完整的 slide
        slide_pattern = re.compile(r'([-^<>vVpqszw]|pp|qq)(\d)(\[.+?\])?')
        matches = list(slide_pattern.finditer(remaining))
        
        if not matches:
            return
        
        # 取最后一个匹配的时值
        duration = None
        for match in reversed(matches):
            if match.group(3):
                duration = self._parse_duration(match.group(3))
                break
        
        # 构建完整路径
        path_parts = []
        current_pos = start_pos
        for match in matches:
            path_type = match.group(1)
            end_pos = match.group(2)
            path_parts.append(f"{current_pos}{path_type}{end_pos}")
            current_pos = end_pos
        
        note = Note(
            time=self.current_time,
            position=start_pos,
            note_type=NoteType.SLIDE,
            is_break=is_break,
            is_ex=is_ex,
            slide_path='/'.join(path_parts) if path_parts else remaining,
            slide_end=current_pos,
            slide_duration=duration,
            is_star=True,  # Slide 起点默认是星星
            star_rotating=star_rotating,
            is_fake_each=is_fake_each,
            raw=raw
        )
        self.notes.append(note)
    
    def _parse_duration(self, text: str) -> Optional[Duration]:
        """解析时值"""
        # [X:Y] - Y 个 X 分音符
        # [#X:Y] - X 秒 * Y
        # [#X] - X 秒
        # [BPM#X:Y] - 使用指定 BPM
        
        match = re.search(r'\[(?:(\d+(?:\.\d+)?)\#)?(\d+)(?::(\d+))?\]', text)
        if match:
            bpm_override = float(match.group(1)) if match.group(1) else self.current_bpm
            divisor = int(match.group(2))
            count = int(match.group(3)) if match.group(3) else 1
            return Duration.from_beat(divisor, count, bpm_override)
        
        match = re.search(r'\[\#(\d+(?:\.\d+)?)(?::(\d+(?:\.\d+)?))?\]', text)
        if match:
            seconds = float(match.group(1))
            multiplier = float(match.group(2)) if match.group(2) else 1
            return Duration.from_seconds(seconds * multiplier)
        
        return None


class SimaiGenerator:
    """Simai 格式生成器"""
    
    def __init__(self, bpm: float = 120.0, divisor: int = 4):
        self.bpm = bpm
        self.divisor = divisor
        self.time_per_beat = 60.0 / bpm * 4 / divisor
        
    def generate(self, notes: List[Note], meta: ChartMeta) -> str:
        """生成完整的 simai 文本"""
        # 按时间排序
        sorted_notes = sorted(notes, key=lambda n: n.time)
        
        # 生成文件头
        header = self._generate_header(meta)
        
        # 生成谱面数据
        chart_data = self._generate_chart_data(sorted_notes, meta)
        
        return header + chart_data
    
    def _generate_header(self, meta: ChartMeta) -> str:
        """生成文件头"""
        lines = [
            f"&title={meta.title}",
            f"&wholebpm={meta.bpm}",
            f"&first={meta.offset}",
            f"&lv_5={meta.level}",
            f"&des_5={meta.designer}",
            "&inote_5="
        ]
        return '\n'.join(lines) + '\n'
    
    def _generate_chart_data(self, notes: List[Note], meta: ChartMeta) -> str:
        """生成谱面主体"""
        if not notes:
            return ""
        
        result = []
        result.append(f"({meta.bpm})")
        result.append(f"{{{self.divisor}}}")
        
        # 将音符按时间分组
        time_groups: Dict[float, List[Note]] = {}
        for note in notes:
            # 量化到最近的网格点
            quantized_time = round(note.time / self.time_per_beat) * self.time_per_beat
            if quantized_time not in time_groups:
                time_groups[quantized_time] = []
            time_groups[quantized_time].append(note)
        
        # 按时间顺序生成
        sorted_times = sorted(time_groups.keys())
        current_time = meta.offset
        
        for target_time in sorted_times:
            # 插入逗号推进时间
            while current_time < target_time - self.time_per_beat / 2:
                result.append(",")
                current_time += self.time_per_beat
            
            # 生成该时间点的所有音符
            note_strs = []
            for note in time_groups[target_time]:
                note_strs.append(self._note_to_string(note))
            
            result.append("/".join(note_strs))
        
        # 结束
        result.append(",E")
        
        return "".join(result)
    
    def _note_to_string(self, note: Note) -> str:
        """将音符转换为 simai 字符串"""
        parts = []
        
        # 前缀
        if note.is_break:
            parts.append("b")
        if note.is_ex:
            parts.append("x")
        
        # 位置
        parts.append(note.position)
        
        # Hold
        if note.note_type in (NoteType.HOLD, NoteType.TOUCH_HOLD):
            if note.hold_duration:
                parts.append(f"h{note.hold_duration.raw}")
            else:
                parts.append("h[4:1]")
        
        # Slide
        elif note.note_type == NoteType.SLIDE:
            if note.slide_path:
                # 简化：只取路径类型和终点
                path_match = re.search(r'[-^<>vVpqszw]|pp|qq', note.slide_path)
                if path_match:
                    parts.append(path_match.group())
                    parts.append(note.slide_end or "5")
                if note.slide_duration:
                    parts.append(note.slide_duration.raw)
        
        # 特殊标记
        if note.is_star and note.note_type == NoteType.TAP:
            if note.star_rotating:
                parts.append("$$")
            else:
                parts.append("$")
        
        if note.is_firework:
            parts.append("f")
        
        return "".join(parts)


def parse_simai(text: str) -> Chart:
    """便捷函数：解析 simai 文本"""
    parser = SimaiParser()
    return parser.parse(text)


def generate_simai(notes: List[Note], meta: ChartMeta, bpm: float = 120.0, divisor: int = 4) -> str:
    """便捷函数：生成 simai 文本"""
    generator = SimaiGenerator(bpm, divisor)
    return generator.generate(notes, meta)


if __name__ == "__main__":
    # 测试解析器
    test_chart = """
&title=Test Song
&wholebpm=150
&first=0.5
&lv_5=12+
&des_5=AI
&inote_5=
(150){4}
1,2,3,4,5,6,7,8,
1/5,2/6,3/7,4/8,
1h[4:2],5h[4:2],,
1-5[4:1],
b3,bx7,
A1,B2f,C,
E
"""
    chart = parse_simai(test_chart)
    print(f"Title: {chart.meta.title}")
    print(f"BPM: {chart.meta.bpm}")
    print(f"Notes: {len(chart.notes)}")
    for note in chart.notes[:10]:
        print(f"  {note.time:.3f}s: {note.note_type.name} at {note.position}")
