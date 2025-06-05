#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp
import numpy as np

# MediaPipe设置
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

class SimpleFaceTest:
    def __init__(self):
        model_path = "face_landmarker.task"
        
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        
        self.landmarker = FaceLandmarker.create_from_options(options)
        
    def run(self):
        print("开始简单人脸检测测试...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("无法打开摄像头")
            return
            
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
                timestamp_ms = int((frame_count / 30.0) * 1000)
                result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
                
                # 简单绘制
                if result.face_landmarks:
                    for face_landmarks in result.face_landmarks:
                        for landmark in face_landmarks:
                            x = int(landmark.x * frame.shape[1])
                            y = int(landmark.y * frame.shape[0])
                            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                cv2.imshow('Simple Face Test', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                    
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("测试完成")

if __name__ == "__main__":
    test = SimpleFaceTest()
    test.run() 