#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MediaPipe Landmarks 转 像素坐标示例
将归一化的landmarks坐标转换为1280x720窗口中的像素位置
"""

import cv2
import mediapipe as mp
import numpy as np

class LandmarksToPixels:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        
        # 窗口尺寸
        self.window_width = 1280
        self.window_height = 720
        
        # 4个关键点的索引（与face_mask_renderer.py一致）
        self.forehead_index = 10    # 额头
        self.left_cheek_index = 234  # 左脸颊
        self.chin_index = 152        # 下巴
        self.right_cheek_index = 454  # 右脸颊
        
        # 创建MediaPipe人脸检测器
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print(f"✅ 初始化完成")
        print(f"📐 窗口尺寸: {self.window_width} x {self.window_height}")
        print(f"🎯 4个关键点索引: {[self.forehead_index, self.left_cheek_index, self.chin_index, self.right_cheek_index]}")
    
    def landmarks_to_pixels(self, landmarks, apply_mirror=True):
        """
        将MediaPipe的归一化landmarks转换为像素坐标
        
        Args:
            landmarks: MediaPipe的landmarks结果
            apply_mirror: 是否应用镜像翻转（通常摄像头需要）
            
        Returns:
            dict: 包含4个关键点像素坐标的字典
        """
        if not landmarks:
            return None
        
        # 提取4个关键点的归一化坐标
        points = {}
        key_indices = {
            'forehead': self.forehead_index,
            'left_cheek': self.left_cheek_index, 
            'chin': self.chin_index,
            'right_cheek': self.right_cheek_index
        }
        
        for name, idx in key_indices.items():
            if idx < len(landmarks.landmark):
                landmark = landmarks.landmark[idx]
                
                # 转换为像素坐标
                x_pixel = landmark.x * self.window_width
                y_pixel = landmark.y * self.window_height
                
                # 可选：应用镜像翻转（摄像头通常是镜像的）
                if apply_mirror:
                    x_pixel = self.window_width - 1 - x_pixel
                
                # 确保坐标在有效范围内
                x_pixel = max(0, min(self.window_width - 1, x_pixel))
                y_pixel = max(0, min(self.window_height - 1, y_pixel))
                
                points[name] = {
                    'pixel': (int(x_pixel), int(y_pixel)),
                    'normalized': (landmark.x, landmark.y, landmark.z),
                    'index': idx
                }
        
        return points
    
    def run_demo(self):
        """运行实时演示"""
        print("\n🎥 启动摄像头演示...")
        print("按 'q' 退出，按 'm' 切换镜像模式")
        
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.window_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.window_height)
        
        apply_mirror = True
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # 原始帧处理
            if apply_mirror:
                frame = cv2.flip(frame, 1)
            
            # 确保帧尺寸正确
            frame = cv2.resize(frame, (self.window_width, self.window_height))
            
            # MediaPipe处理
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            # 转换landmarks为像素坐标
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 获取4个关键点的像素坐标
                    pixel_points = self.landmarks_to_pixels(face_landmarks, apply_mirror=False)
                    
                    if pixel_points:
                        # 绘制关键点
                        colors = {
                            'forehead': (0, 255, 0),      # 绿色
                            'left_cheek': (255, 0, 0),    # 蓝色
                            'chin': (0, 255, 255),        # 黄色
                            'right_cheek': (255, 0, 255)  # 紫色
                        }
                        
                        for name, data in pixel_points.items():
                            x, y = data['pixel']
                            color = colors[name]
                            
                            # 绘制关键点
                            cv2.circle(frame, (x, y), 8, color, -1)
                            
                            # 显示坐标信息
                            text = f"{name}[{data['index']}]: ({x}, {y})"
                            cv2.putText(frame, text, (x + 10, y - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        
                        # 绘制连接线
                        if len(pixel_points) == 4:
                            forehead = pixel_points['forehead']['pixel']
                            left_cheek = pixel_points['left_cheek']['pixel']
                            chin = pixel_points['chin']['pixel']
                            right_cheek = pixel_points['right_cheek']['pixel']
                            
                            # 绘制面部轮廓
                            cv2.line(frame, forehead, left_cheek, (255, 255, 255), 2)
                            cv2.line(frame, left_cheek, chin, (255, 255, 255), 2)
                            cv2.line(frame, chin, right_cheek, (255, 255, 255), 2)
                            cv2.line(frame, right_cheek, forehead, (255, 255, 255), 2)
                            
                            # 计算面部尺寸
                            face_width = abs(right_cheek[0] - left_cheek[0])
                            face_height = abs(chin[1] - forehead[1])
                            
                            # 显示尺寸信息
                            info_text = f"Face Size: {face_width}x{face_height} pixels"
                            cv2.putText(frame, info_text, (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示窗口信息
            window_info = f"Window: {self.window_width}x{self.window_height}, Mirror: {apply_mirror}"
            cv2.putText(frame, window_info, (10, self.window_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示结果
            cv2.imshow("Landmarks to Pixels Demo", frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                apply_mirror = not apply_mirror
                print(f"镜像模式: {'开启' if apply_mirror else '关闭'}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ 演示结束")

def main():
    """主函数"""
    print("MediaPipe Landmarks 转像素坐标工具")
    print("=" * 50)
    
    try:
        converter = LandmarksToPixels(camera_id=0)
        converter.run_demo()
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 