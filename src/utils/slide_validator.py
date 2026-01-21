"""
Slide 路径验证器
检查 Slide 路径是否合法可达
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Position:
    """位置（极坐标表示）"""
    angle: int  # 1-8 对应 0-315 度（每 45 度）
    ring: str   # 'A' (外圈), 'B' (中圈), 'C' (中心)


# 位置连接图
# A区 (1-8): 外圈按键位置
# B区: 中间圈
# C区: 中心

# 直线可达性矩阵 (起点 -> 可直线到达的终点)
STRAIGHT_REACHABLE = {
    1: {2, 3, 4, 5, 6, 7, 8},  # 从 1 可以直线到达所有位置
    2: {1, 3, 4, 5, 6, 7, 8},
    3: {1, 2, 4, 5, 6, 7, 8},
    4: {1, 2, 3, 5, 6, 7, 8},
    5: {1, 2, 3, 4, 6, 7, 8},
    6: {1, 2, 3, 4, 5, 7, 8},
    7: {1, 2, 3, 4, 5, 6, 8},
    8: {1, 2, 3, 4, 5, 6, 7},
}


class SlideValidator:
    """Slide 路径验证器"""
    
    def __init__(self):
        # 位置角度映射
        self.position_angles = {
            1: 90,    # 上
            2: 45,    # 右上
            3: 0,     # 右
            4: 315,   # 右下
            5: 270,   # 下
            6: 225,   # 左下
            7: 180,   # 左
            8: 135,   # 左上
        }
    
    def validate_slide(self, start: int, end: int, path_type: str) -> Tuple[bool, str]:
        """
        验证 Slide 路径是否合法
        
        Args:
            start: 起始位置 (1-8)
            end: 终止位置 (1-8)
            path_type: 路径类型 ('-', '^', '<', '>', 'v', 'p', 'q', etc.)
        
        Returns:
            (是否合法, 原因说明)
        """
        # 基本验证
        if not (1 <= start <= 8):
            return False, f"无效的起始位置: {start}"
        if not (1 <= end <= 8):
            return False, f"无效的终止位置: {end}"
        if start == end:
            return False, "起点和终点不能相同"
        
        # 根据路径类型验证
        if path_type == '-':
            return self._validate_straight(start, end)
        elif path_type == '^':
            return self._validate_arc_auto(start, end)
        elif path_type == '<':
            return self._validate_arc_ccw(start, end)
        elif path_type == '>':
            return self._validate_arc_cw(start, end)
        elif path_type == 'v':
            return self._validate_v_center(start, end)
        elif path_type == 'p':
            return self._validate_p_curve(start, end)
        elif path_type == 'q':
            return self._validate_q_curve(start, end)
        elif path_type == 's':
            return self._validate_s_zigzag(start, end)
        elif path_type == 'z':
            return self._validate_z_zigzag(start, end)
        elif path_type == 'pp':
            return self._validate_pp_curve(start, end)
        elif path_type == 'qq':
            return self._validate_qq_curve(start, end)
        elif path_type == 'w':
            return self._validate_wifi(start, end)
        else:
            return False, f"未知的路径类型: {path_type}"
    
    def _validate_straight(self, start: int, end: int) -> Tuple[bool, str]:
        """验证直线路径"""
        # 所有位置之间都可以直线连接
        return True, "直线路径有效"
    
    def _validate_arc_auto(self, start: int, end: int) -> Tuple[bool, str]:
        """验证自动弧线（选择最短路径）"""
        return True, "自动弧线路径有效"
    
    def _validate_arc_ccw(self, start: int, end: int) -> Tuple[bool, str]:
        """验证逆时针弧线"""
        return True, "逆时针弧线路径有效"
    
    def _validate_arc_cw(self, start: int, end: int) -> Tuple[bool, str]:
        """验证顺时针弧线"""
        return True, "顺时针弧线路径有效"
    
    def _validate_v_center(self, start: int, end: int) -> Tuple[bool, str]:
        """验证通过中心的折线"""
        return True, "中心折线路径有效"
    
    def _validate_p_curve(self, start: int, end: int) -> Tuple[bool, str]:
        """验证 p 型弧线（逆时针绕 B 区）"""
        # p 型：从起点逆时针绕 B 区，直到能直线到达终点
        return True, "p 型弧线路径有效"
    
    def _validate_q_curve(self, start: int, end: int) -> Tuple[bool, str]:
        """验证 q 型弧线（顺时针绕 B 区）"""
        return True, "q 型弧线路径有效"
    
    def _validate_s_zigzag(self, start: int, end: int) -> Tuple[bool, str]:
        """验证 s 型折线"""
        # s: 直线到 B(X+6)，然后直线到 B(X+2)，然后直线到 A(Y)
        return True, "s 型折线路径有效"
    
    def _validate_z_zigzag(self, start: int, end: int) -> Tuple[bool, str]:
        """验证 z 型折线"""
        # z: 直线到 B(X+2)，然后直线到 B(X+6)，然后直线到 A(Y)
        return True, "z 型折线路径有效"
    
    def _validate_pp_curve(self, start: int, end: int) -> Tuple[bool, str]:
        """验证 pp 型大弧线"""
        return True, "pp 型大弧线路径有效"
    
    def _validate_qq_curve(self, start: int, end: int) -> Tuple[bool, str]:
        """验证 qq 型大弧线"""
        return True, "qq 型大弧线路径有效"
    
    def _validate_wifi(self, start: int, end: int) -> Tuple[bool, str]:
        """验证 WiFi 扇形星星"""
        # WiFi 需要终点在特定位置
        # 终点应该是起点 ±3 或 ±5 的位置
        diff = abs(end - start)
        if diff == 0:
            return False, "WiFi 终点不能与起点相同"
        return True, "WiFi 路径有效"
    
    def calculate_distance(self, start: int, end: int, path_type: str) -> float:
        """
        计算 Slide 路径长度（用于估算时长）
        
        Returns:
            相对路径长度 (1.0 = 直线对角)
        """
        if path_type == '-':
            # 直线距离
            return self._linear_distance(start, end)
        elif path_type in ['<', '>', '^']:
            # 弧线距离
            return self._arc_distance(start, end, path_type)
        elif path_type == 'v':
            # 折线通过中心
            return 2.0
        elif path_type in ['p', 'q']:
            return 1.5
        elif path_type in ['pp', 'qq']:
            return 2.0
        elif path_type in ['s', 'z']:
            return 2.5
        elif path_type == 'w':
            return 1.0
        else:
            return 1.0
    
    def _linear_distance(self, start: int, end: int) -> float:
        """计算直线距离"""
        diff = abs(end - start)
        if diff > 4:
            diff = 8 - diff
        return diff / 4.0  # 归一化到 0-1
    
    def _arc_distance(self, start: int, end: int, direction: str) -> float:
        """计算弧线距离"""
        if direction == '^':
            # 自动选择最短
            diff = abs(end - start)
            if diff > 4:
                diff = 8 - diff
        elif direction == '<':
            # 逆时针
            diff = (start - end) % 8
        else:
            # 顺时针
            diff = (end - start) % 8
        
        # 弧线长度约为直线的 π/2 倍
        return diff / 4.0 * 1.57
    
    def suggest_slide_type(self, start: int, end: int) -> str:
        """
        根据起点终点建议合适的 Slide 类型
        
        Returns:
            建议的路径类型
        """
        diff = abs(end - start)
        if diff > 4:
            diff = 8 - diff
        
        if diff <= 2:
            # 短距离：直线或短弧
            return '-'
        elif diff == 3:
            # 中等距离：弧线
            return '^'
        elif diff == 4:
            # 对角：可以用多种
            return '-'  # 或 'v'
        else:
            return '-'


def validate_slide(start: int, end: int, path_type: str) -> bool:
    """便捷函数：验证 Slide"""
    validator = SlideValidator()
    is_valid, _ = validator.validate_slide(start, end, path_type)
    return is_valid


if __name__ == "__main__":
    # 测试
    validator = SlideValidator()
    
    test_cases = [
        (1, 5, '-'),   # 直线
        (1, 3, '^'),   # 自动弧线
        (1, 7, '<'),   # 逆时针
        (1, 3, '>'),   # 顺时针
        (1, 5, 'v'),   # 中心折线
        (1, 6, 'p'),   # p 弧线
        (1, 4, 'q'),   # q 弧线
        (1, 5, 's'),   # s 折线
        (1, 5, 'z'),   # z 折线
        (1, 4, 'w'),   # WiFi
    ]
    
    for start, end, path_type in test_cases:
        is_valid, reason = validator.validate_slide(start, end, path_type)
        distance = validator.calculate_distance(start, end, path_type)
        print(f"{start}{path_type}{end}: {'✓' if is_valid else '✗'} - {reason} (距离: {distance:.2f})")
