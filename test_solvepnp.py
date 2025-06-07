#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
solvePnP测试脚本
快速验证重投影误差改善情况
"""

import cv2
import mediapipe as mp
import numpy as np
import time

# MediaPipe 导入
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def load_obj_vertices(path):
    """加载OBJ文件顶点"""
    vertices = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append([x, y, z])
    return np.array(vertices, dtype=np.float32)

def load_camera_calibration():
    """加载相机标定参数"""
    try:
        if os.path.exists("calib.npz"):
            calib_data = np.load("calib.npz")
            return calib_data['K'], calib_data['dist']
        else:
            print("标定文件不存在，使用估计参数")
            K = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float32)
            dist = np.zeros((5,), dtype=np.float64)
            return K, dist
    except:
        print("加载标定失败，使用估计参数")
        K = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float32)
        dist = np.zeros((5,), dtype=np.float64)
        return K, dist

def test_solvepnp():
    """测试solvePnP实现"""
    print("=== solvePnP测试开始 ===")
    
    # 加载模型和标定
    try:
        vertices = load_obj_vertices("obj/Andy_Wah_facemesh.obj")
        print(f"✅ 模型加载成功: {len(vertices)} 个顶点")
        
        # 模型尺寸分析
        bbox = np.array([
            [vertices[:, 0].min(), vertices[:, 1].min(), vertices[:, 2].min()],
            [vertices[:, 0].max(), vertices[:, 1].max(), vertices[:, 2].max()]
        ])
        width = bbox[1, 0] - bbox[0, 0]
        height = bbox[1, 1] - bbox[0, 1]
        depth = bbox[1, 2] - bbox[0, 2]
        
        print(f"📏 模型尺寸: 宽{width:.2f} 高{height:.2f} 深{depth:.2f}")
        
        # 缩放到真实人脸尺寸
        target_width = 160.0  # 毫米
        scale_factor = target_width / width
        vertices_scaled = vertices * scale_factor
        
        print(f"🔧 缩放系数: {scale_factor:.3f}x (目标宽度{target_width}mm)")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 加载相机标定
    K, dist = load_camera_calibration()
    print("📷 相机参数:")
    print(f"   fx={K[0,0]:.2f}, fy={K[1,1]:.2f}")
    print(f"   cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")
    
    # 关键点索引
    pnp_indices = [1, 168, 10, 152, 175, 33, 263, 130, 359, 70, 300, 107, 336, 19, 94, 234, 454, 172, 397]
    
    # 创建MediaPipe检测器
    model_path = "face_landmarker.task"
    if not os.path.exists(model_path):
        print("❌ MediaPipe模型不存在")
        return
    
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1
    )
    
    try:
        landmarker = FaceLandmarker.create_from_options(options)
        print("✅ MediaPipe检测器创建成功")
    except Exception as e:
        print(f"❌ MediaPipe检测器创建失败: {e}")
        return
    
    # 开始摄像头测试
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return
    
    print("🎥 开始测试...")
    print("按ESC退出，按空格显示详细信息")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_count += 1
            
            # 转换为RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # 检测
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                
                # 准备solvePnP数据
                img_points = []
                obj_points = []
                
                for idx in pnp_indices:
                    if idx < len(landmarks) and idx < len(vertices_scaled):
                        x_pixel = landmarks[idx].x * frame.shape[1]
                        y_pixel = landmarks[idx].y * frame.shape[0]
                        img_points.append([x_pixel, y_pixel])
                        obj_points.append(vertices_scaled[idx])
                
                if len(img_points) >= 8:
                    img_points = np.array(img_points, dtype=np.float32)
                    obj_points = np.array(obj_points, dtype=np.float32)
                    
                    # solvePnP
                    try:
                        success, rvec, tvec = cv2.solvePnP(
                            obj_points, img_points, K, dist,
                            flags=cv2.SOLVEPNP_ITERATIVE
                        )
                        
                        if success:
                            # 计算重投影误差
                            reproj_points, _ = cv2.projectPoints(obj_points, rvec, tvec, K, dist)
                            reproj_points = reproj_points.reshape(-1, 2)
                            reproj_error = np.mean(np.linalg.norm(img_points - reproj_points, axis=1))
                            
                            # 显示结果
                            if reproj_error < 10:
                                color = (0, 255, 0)  # 绿色：优秀
                                status = "优秀"
                            elif reproj_error < 20:
                                color = (0, 255, 255)  # 黄色：良好
                                status = "良好"
                            else:
                                color = (0, 0, 255)  # 红色：需改进
                                status = "需改进"
                            
                            cv2.putText(frame, f"Reproj Error: {reproj_error:.2f}px ({status})", 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                            
                            # 每30帧输出详细信息
                            if frame_count % 30 == 0:
                                print(f"第{frame_count}帧 - 重投影误差: {reproj_error:.2f}px ({status})")
                                print(f"  平移: T=[{tvec[0,0]:.1f}, {tvec[1,0]:.1f}, {tvec[2,0]:.1f}]mm")
                        else:
                            cv2.putText(frame, "solvePnP Failed", (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    except Exception as e:
                        cv2.putText(frame, f"Error: {str(e)[:30]}", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # 绘制关键点
                for idx in pnp_indices:
                    if idx < len(landmarks):
                        x = int(landmarks[idx].x * frame.shape[1])
                        y = int(landmarks[idx].y * frame.shape[0])
                        cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
            
            cv2.imshow('solvePnP测试', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("=== 测试结束 ===")

if __name__ == "__main__":
    import os
    test_solvepnp() 