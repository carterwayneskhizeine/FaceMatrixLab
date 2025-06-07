#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FaceMatrixLab 表情驱动系统
提供基于 MediaPipe landmarks 的高级表情驱动功能
作为 BlendShapes 的补充或备用方案
"""

import numpy as np
import cv2

class AdvancedExpressionDriver:
    def __init__(self):
        """初始化高级表情驱动系统"""
        print("🎭 初始化高级表情驱动系统...")
        
        # 表情检测阈值
        self.mouth_open_threshold = 0.02
        self.smile_threshold = 0.03
        self.eye_blink_threshold = 0.8
        self.eyebrow_raise_threshold = 0.02
        
        # 表情强度系数
        self.mouth_sensitivity = 2.0
        self.eye_sensitivity = 1.5
        self.eyebrow_sensitivity = 1.8
        
        # 中性状态缓存
        self.neutral_mouth_height = None
        self.neutral_eye_heights = {}
        self.neutral_eyebrow_heights = {}
        self.neutral_mouth_width = None
        
        # 表情状态缓存（平滑处理）
        self.prev_mouth_open = 0.0
        self.prev_smile_level = 0.0
        self.prev_eye_blink = {'left': 0.0, 'right': 0.0}
        self.prev_eyebrow_raise = {'left': 0.0, 'right': 0.0}
        
        # 平滑系数
        self.smoothing_factor = 0.7
        
        print("✅ 高级表情驱动系统初始化完成")
    
    def calibrate_neutral_expression(self, landmarks_list):
        """校准中性表情状态（使用多帧平均）"""
        if len(landmarks_list) < 10:
            return False
        
        print("🔧 校准中性表情状态...")
        
        mouth_heights = []
        mouth_widths = []
        left_eye_heights = []
        right_eye_heights = []
        left_eyebrow_heights = []
        right_eyebrow_heights = []
        
        for landmarks in landmarks_list:
            # 嘴巴高度和宽度
            mouth_top = np.array([landmarks[13].x, landmarks[13].y])  # 上唇中央
            mouth_bottom = np.array([landmarks[14].x, landmarks[14].y])  # 下唇中央
            mouth_left = np.array([landmarks[61].x, landmarks[61].y])  # 左嘴角
            mouth_right = np.array([landmarks[291].x, landmarks[291].y])  # 右嘴角
            
            mouth_height = np.linalg.norm(mouth_bottom - mouth_top)
            mouth_width = np.linalg.norm(mouth_right - mouth_left)
            mouth_heights.append(mouth_height)
            mouth_widths.append(mouth_width)
            
            # 眼睛高度
            # 左眼
            left_eye_top = np.array([landmarks[159].x, landmarks[159].y])
            left_eye_bottom = np.array([landmarks[145].x, landmarks[145].y])
            left_eye_height = np.linalg.norm(left_eye_bottom - left_eye_top)
            left_eye_heights.append(left_eye_height)
            
            # 右眼
            right_eye_top = np.array([landmarks[386].x, landmarks[386].y])
            right_eye_bottom = np.array([landmarks[374].x, landmarks[374].y])
            right_eye_height = np.linalg.norm(right_eye_bottom - right_eye_top)
            right_eye_heights.append(right_eye_height)
            
            # 眉毛高度
            # 左眉
            left_eyebrow = np.array([landmarks[70].x, landmarks[70].y])
            left_eye_center = np.array([landmarks[33].x, landmarks[33].y])
            left_eyebrow_height = np.linalg.norm(left_eyebrow - left_eye_center)
            left_eyebrow_heights.append(left_eyebrow_height)
            
            # 右眉
            right_eyebrow = np.array([landmarks[300].x, landmarks[300].y])
            right_eye_center = np.array([landmarks[362].x, landmarks[362].y])
            right_eyebrow_height = np.linalg.norm(right_eyebrow - right_eye_center)
            right_eyebrow_heights.append(right_eyebrow_height)
        
        # 计算平均值作为中性状态
        self.neutral_mouth_height = np.mean(mouth_heights)
        self.neutral_mouth_width = np.mean(mouth_widths)
        self.neutral_eye_heights['left'] = np.mean(left_eye_heights)
        self.neutral_eye_heights['right'] = np.mean(right_eye_heights)
        self.neutral_eyebrow_heights['left'] = np.mean(left_eyebrow_heights)
        self.neutral_eyebrow_heights['right'] = np.mean(right_eyebrow_heights)
        
        print(f"✅ 中性状态校准完成:")
        print(f"   嘴巴高度: {self.neutral_mouth_height:.4f}")
        print(f"   嘴巴宽度: {self.neutral_mouth_width:.4f}")
        print(f"   左眼高度: {self.neutral_eye_heights['left']:.4f}")
        print(f"   右眼高度: {self.neutral_eye_heights['right']:.4f}")
        
        return True
    
    def detect_mouth_expression(self, landmarks):
        """检测嘴巴表情"""
        if self.neutral_mouth_height is None:
            return {'open': 0.0, 'smile': 0.0}
        
        # 嘴巴高度
        mouth_top = np.array([landmarks[13].x, landmarks[13].y])
        mouth_bottom = np.array([landmarks[14].x, landmarks[14].y])
        current_mouth_height = np.linalg.norm(mouth_bottom - mouth_top)
        
        # 嘴巴宽度
        mouth_left = np.array([landmarks[61].x, landmarks[61].y])
        mouth_right = np.array([landmarks[291].x, landmarks[291].y])
        current_mouth_width = np.linalg.norm(mouth_right - mouth_left)
        
        # 嘴角位置（用于检测微笑）
        mouth_corner_left = np.array([landmarks[61].x, landmarks[61].y])
        mouth_corner_right = np.array([landmarks[291].x, landmarks[291].y])
        mouth_center = (mouth_corner_left + mouth_corner_right) / 2
        
        # 计算嘴巴张开程度
        mouth_open_ratio = (current_mouth_height - self.neutral_mouth_height) / self.neutral_mouth_height
        mouth_open = max(0, mouth_open_ratio * self.mouth_sensitivity)
        
        # 计算微笑程度（基于嘴角上升和宽度变化）
        width_ratio = (current_mouth_width - self.neutral_mouth_width) / self.neutral_mouth_width
        
        # 嘴角相对于基准线的高度
        baseline_y = mouth_center[1]
        corner_lift_left = baseline_y - mouth_corner_left[1]
        corner_lift_right = baseline_y - mouth_corner_right[1]
        avg_corner_lift = (corner_lift_left + corner_lift_right) / 2
        
        smile_level = max(0, (width_ratio + avg_corner_lift * 10) * self.mouth_sensitivity)
        
        # 平滑处理
        mouth_open = self.prev_mouth_open * self.smoothing_factor + mouth_open * (1 - self.smoothing_factor)
        smile_level = self.prev_smile_level * self.smoothing_factor + smile_level * (1 - self.smoothing_factor)
        
        self.prev_mouth_open = mouth_open
        self.prev_smile_level = smile_level
        
        return {
            'open': np.clip(mouth_open, 0, 2.0),
            'smile': np.clip(smile_level, 0, 1.5)
        }
    
    def detect_eye_expression(self, landmarks):
        """检测眼睛表情"""
        if not self.neutral_eye_heights:
            return {'left_blink': 0.0, 'right_blink': 0.0}
        
        # 左眼
        left_eye_top = np.array([landmarks[159].x, landmarks[159].y])
        left_eye_bottom = np.array([landmarks[145].x, landmarks[145].y])
        current_left_height = np.linalg.norm(left_eye_bottom - left_eye_top)
        
        # 右眼
        right_eye_top = np.array([landmarks[386].x, landmarks[386].y])
        right_eye_bottom = np.array([landmarks[374].x, landmarks[374].y])
        current_right_height = np.linalg.norm(right_eye_bottom - right_eye_top)
        
        # 计算眨眼程度（高度减少表示闭合）
        left_blink_ratio = 1.0 - (current_left_height / self.neutral_eye_heights['left'])
        right_blink_ratio = 1.0 - (current_right_height / self.neutral_eye_heights['right'])
        
        left_blink = max(0, left_blink_ratio * self.eye_sensitivity)
        right_blink = max(0, right_blink_ratio * self.eye_sensitivity)
        
        # 平滑处理
        left_blink = self.prev_eye_blink['left'] * self.smoothing_factor + left_blink * (1 - self.smoothing_factor)
        right_blink = self.prev_eye_blink['right'] * self.smoothing_factor + right_blink * (1 - self.smoothing_factor)
        
        self.prev_eye_blink['left'] = left_blink
        self.prev_eye_blink['right'] = right_blink
        
        return {
            'left_blink': np.clip(left_blink, 0, 1.0),
            'right_blink': np.clip(right_blink, 0, 1.0)
        }
    
    def detect_eyebrow_expression(self, landmarks):
        """检测眉毛表情"""
        if not self.neutral_eyebrow_heights:
            return {'left_raise': 0.0, 'right_raise': 0.0}
        
        # 左眉
        left_eyebrow = np.array([landmarks[70].x, landmarks[70].y])
        left_eye_center = np.array([landmarks[33].x, landmarks[33].y])
        current_left_height = np.linalg.norm(left_eyebrow - left_eye_center)
        
        # 右眉
        right_eyebrow = np.array([landmarks[300].x, landmarks[300].y])
        right_eye_center = np.array([landmarks[362].x, landmarks[362].y])
        current_right_height = np.linalg.norm(right_eyebrow - right_eye_center)
        
        # 计算眉毛上扬程度
        left_raise_ratio = (current_left_height - self.neutral_eyebrow_heights['left']) / self.neutral_eyebrow_heights['left']
        right_raise_ratio = (current_right_height - self.neutral_eyebrow_heights['right']) / self.neutral_eyebrow_heights['right']
        
        left_raise = max(0, left_raise_ratio * self.eyebrow_sensitivity)
        right_raise = max(0, right_raise_ratio * self.eyebrow_sensitivity)
        
        # 平滑处理
        left_raise = self.prev_eyebrow_raise['left'] * self.smoothing_factor + left_raise * (1 - self.smoothing_factor)
        right_raise = self.prev_eyebrow_raise['right'] * self.smoothing_factor + right_raise * (1 - self.smoothing_factor)
        
        self.prev_eyebrow_raise['left'] = left_raise
        self.prev_eyebrow_raise['right'] = right_raise
        
        return {
            'left_raise': np.clip(left_raise, 0, 1.5),
            'right_raise': np.clip(right_raise, 0, 1.5)
        }
    
    def get_full_expression_data(self, landmarks):
        """获取完整的表情数据"""
        mouth_expr = self.detect_mouth_expression(landmarks)
        eye_expr = self.detect_eye_expression(landmarks)
        eyebrow_expr = self.detect_eyebrow_expression(landmarks)
        
        return {
            'mouth': mouth_expr,
            'eyes': eye_expr,
            'eyebrows': eyebrow_expr
        }
    
    def apply_expression_to_vertices(self, vertices, vertex_groups, expression_data, strength=1.0):
        """将表情数据应用到顶点"""
        vertex_offsets = np.zeros_like(vertices)
        
        # 应用嘴巴表情
        mouth_data = expression_data['mouth']
        if 'mouth' in vertex_groups:
            for idx in vertex_groups['mouth']:
                if idx < len(vertex_offsets):
                    # 嘴巴张开
                    vertex_offsets[idx, 1] -= mouth_data['open'] * strength * 3.0
                    # 微笑
                    vertex_offsets[idx, 1] += mouth_data['smile'] * strength * 1.5
                    vertex_offsets[idx, 0] += mouth_data['smile'] * strength * 1.0 * np.sign(vertices[idx, 0])
        
        # 应用眼睛表情
        eye_data = expression_data['eyes']
        if 'left_eye' in vertex_groups:
            for idx in vertex_groups['left_eye']:
                if idx < len(vertex_offsets):
                    vertex_offsets[idx, 1] -= eye_data['left_blink'] * strength * 1.0
        
        if 'right_eye' in vertex_groups:
            for idx in vertex_groups['right_eye']:
                if idx < len(vertex_offsets):
                    vertex_offsets[idx, 1] -= eye_data['right_blink'] * strength * 1.0
        
        # 应用眉毛表情
        eyebrow_data = expression_data['eyebrows']
        if 'left_eyebrow' in vertex_groups:
            for idx in vertex_groups['left_eyebrow']:
                if idx < len(vertex_offsets):
                    vertex_offsets[idx, 1] += eyebrow_data['left_raise'] * strength * 2.0
        
        if 'right_eyebrow' in vertex_groups:
            for idx in vertex_groups['right_eyebrow']:
                if idx < len(vertex_offsets):
                    vertex_offsets[idx, 1] += eyebrow_data['right_raise'] * strength * 2.0
        
        return vertices + vertex_offsets 