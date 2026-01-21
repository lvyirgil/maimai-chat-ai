"""
单元测试 - Simai 解析器
"""

import pytest
import sys
sys.path.insert(0, '..')

from src.parser import parse_simai, generate_simai, Note, NoteType, ChartMeta


class TestSimaiParser:
    """测试 Simai 解析器"""
    
    def test_parse_basic_tap(self):
        """测试解析基本 TAP 音符"""
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
        
        assert chart.meta.title == "Test"
        assert chart.meta.bpm == 120
        assert len(chart.notes) == 4
        
        # 检查位置
        positions = [n.position for n in chart.notes]
        assert positions == ['1', '2', '3', '4']
        
        # 检查类型
        for note in chart.notes:
            assert note.note_type == NoteType.TAP
    
    def test_parse_hold(self):
        """测试解析 HOLD 音符"""
        chart_text = """
&title=Test
&wholebpm=120
&first=0
&inote_5=
(120){4}
1h[4:2],
E
"""
        chart = parse_simai(chart_text)
        
        assert len(chart.notes) == 1
        note = chart.notes[0]
        assert note.note_type == NoteType.HOLD
        assert note.position == '1'
        assert note.hold_duration is not None
    
    def test_parse_slide(self):
        """测试解析 SLIDE 音符"""
        chart_text = """
&title=Test
&wholebpm=120
&first=0
&inote_5=
(120){4}
1-5[4:1],
E
"""
        chart = parse_simai(chart_text)
        
        assert len(chart.notes) == 1
        note = chart.notes[0]
        assert note.note_type == NoteType.SLIDE
        assert note.position == '1'
        assert note.slide_end == '5'
    
    def test_parse_each(self):
        """测试解析多押"""
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
        
        assert len(chart.notes) == 2
        positions = {n.position for n in chart.notes}
        assert positions == {'1', '5'}
        
        # 时间应该相同
        times = {n.time for n in chart.notes}
        assert len(times) == 1
    
    def test_parse_break(self):
        """测试解析 BREAK 音符"""
        chart_text = """
&title=Test
&wholebpm=120
&first=0
&inote_5=
(120){4}
b1,
E
"""
        chart = parse_simai(chart_text)
        
        assert len(chart.notes) == 1
        note = chart.notes[0]
        assert note.is_break == True
    
    def test_parse_ex(self):
        """测试解析 EX 音符"""
        chart_text = """
&title=Test
&wholebpm=120
&first=0
&inote_5=
(120){4}
x1,
E
"""
        chart = parse_simai(chart_text)
        
        assert len(chart.notes) == 1
        note = chart.notes[0]
        assert note.is_ex == True
    
    def test_parse_touch(self):
        """测试解析 TOUCH 音符"""
        chart_text = """
&title=Test
&wholebpm=120
&first=0
&inote_5=
(120){4}
A1,B2,C,
E
"""
        chart = parse_simai(chart_text)
        
        assert len(chart.notes) == 3
        positions = [n.position for n in chart.notes]
        assert 'A1' in positions
        assert 'B2' in positions
        assert 'C' in positions


class TestSimaiGenerator:
    """测试 Simai 生成器"""
    
    def test_generate_basic(self):
        """测试基本生成"""
        meta = ChartMeta(
            title="Test",
            bpm=120,
            offset=0,
            level="10",
            designer="Test"
        )
        
        notes = [
            Note(time=0, position='1', note_type=NoteType.TAP),
            Note(time=0.5, position='2', note_type=NoteType.TAP),
        ]
        
        simai = generate_simai(notes, meta, bpm=120, divisor=4)
        
        assert "&title=Test" in simai
        assert "&wholebpm=120" in simai
        assert "1" in simai
        assert "2" in simai
    
    def test_roundtrip(self):
        """测试解析-生成循环"""
        original = """
&title=Roundtrip Test
&wholebpm=150
&first=0.5
&lv_5=12
&des_5=Test
&inote_5=
(150){4}
1,2,3,4,
5,6,7,8,
E
"""
        # 解析
        chart = parse_simai(original)
        
        # 重新生成
        regenerated = generate_simai(chart.notes, chart.meta, chart.meta.bpm)
        
        # 再次解析
        chart2 = parse_simai(regenerated)
        
        # 比较
        assert chart2.meta.bpm == chart.meta.bpm
        assert len(chart2.notes) == len(chart.notes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
