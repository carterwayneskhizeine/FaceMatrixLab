#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简洁的MediaPipe landmarks到像素坐标转换工具
"""

def landmarks_to_pixels(landmarks, window_width=1280, window_height=720, 
                       key_indices=None, apply_mirror=False):
    """
    将MediaPipe的归一化landmarks转换为像素坐标
    
    Args:
        landmarks: MediaPipe的face_landmarks结果
        window_width: 窗口宽度（默认1280）
        window_height: 窗口高度（默认720）
        key_indices: 要转换的关键点索引列表，如果为None则转换所有点
        apply_mirror: 是否应用镜像翻转
        
    Returns:
        list: 像素坐标列表 [(x, y), ...]
        或 dict: 如果指定了关键点名称
    """
    if not landmarks or not landmarks.landmark:
        return None
    
    pixel_coords = []
    
    # 如果没有指定关键点，转换所有landmarks
    if key_indices is None:
        indices_to_convert = range(len(landmarks.landmark))
    elif isinstance(key_indices, dict):
        # 如果是字典格式，返回字典结果
        result = {}
        for name, idx in key_indices.items():
            if idx < len(landmarks.landmark):
                landmark = landmarks.landmark[idx]
                x_pixel = landmark.x * window_width
                y_pixel = landmark.y * window_height
                
                if apply_mirror:
                    x_pixel = window_width - 1 - x_pixel
                
                # 边界检查
                x_pixel = max(0, min(window_width - 1, x_pixel))
                y_pixel = max(0, min(window_height - 1, y_pixel))
                
                result[name] = (int(x_pixel), int(y_pixel))
        return result
    else:
        indices_to_convert = key_indices
    
    # 转换指定的索引
    for idx in indices_to_convert:
        if idx < len(landmarks.landmark):
            landmark = landmarks.landmark[idx]
            
            # 基本转换公式
            x_pixel = landmark.x * window_width
            y_pixel = landmark.y * window_height
            
            # 可选镜像翻转
            if apply_mirror:
                x_pixel = window_width - 1 - x_pixel
            
            # 边界检查
            x_pixel = max(0, min(window_width - 1, x_pixel))
            y_pixel = max(0, min(window_height - 1, y_pixel))
            
            pixel_coords.append((int(x_pixel), int(y_pixel)))
    
    return pixel_coords

def get_face_key_points_pixels(landmarks, window_width=1280, window_height=720, 
                              apply_mirror=False):
    """
    获取面部4个关键点的像素坐标
    
    Returns:
        dict: {'forehead': (x,y), 'left_cheek': (x,y), 'chin': (x,y), 'right_cheek': (x,y)}
    """
    key_indices = {
        'forehead': 10,      # 额头
        'left_cheek': 234,   # 左脸颊
        'chin': 152,         # 下巴
        'right_cheek': 454   # 右脸颊
    }
    
    return landmarks_to_pixels(landmarks, window_width, window_height, 
                              key_indices, apply_mirror)

# 使用示例
if __name__ == "__main__":
    print("MediaPipe Landmarks 转换工具函数")
    print("使用示例:")
    print("""
    import mediapipe as mp
    from simple_landmarks_converter import get_face_key_points_pixels
    
    # 初始化MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    
    # 处理图像后...
    results = face_mesh.process(image)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 获取4个关键点的像素坐标
            key_points = get_face_key_points_pixels(face_landmarks)
            
            if key_points:
                print(f"额头坐标: {key_points['forehead']}")
                print(f"左脸颊坐标: {key_points['left_cheek']}")
                print(f"下巴坐标: {key_points['chin']}")
                print(f"右脸颊坐标: {key_points['right_cheek']}")
    """) 