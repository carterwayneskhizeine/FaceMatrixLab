#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FaceMatrixLab 3D 渲染器
使用 Open3D 渲染 Andy_Wah_facemesh.obj，通过 MediaPipe 数据流驱动
支持实时人脸姿态和表情变化
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
import threading
import queue
from typing import Optional, Tuple
import open3d as o3d

# MediaPipe 导入
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


class FaceMatrixLabRenderer:
    def __init__(self, camera_id=0, model_path="../obj/Andy_Wah_facemesh.obj"):
        """初始化3D人脸渲染器"""
        print("=== FaceMatrixLab 3D 渲染器初始化 ===")
        
        # 基本参数
        self.camera_id = camera_id
        self.model_path = model_path
        self.is_running = False
        
        # 渲染参数
        self.render_width = 1280
        self.render_height = 720
        self.fps_target = 30
        
        # MediaPipe 相关
        self.landmarker = None
        self.mp_model_path = self.download_mediapipe_model()
        
        # 数据队列 - 用于线程间通信
        self.data_queue = queue.Queue(maxsize=5)
        self.latest_result = None
        
        # 相机参数（50mm 等效焦距）
        self.setup_camera_parameters()
        
        # 加载3D模型
        if not self.load_face_model():
            raise Exception("无法加载3D模型文件")
        
        # 性能统计
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_render_fps = 0
        self.current_detection_fps = 0
        
        print("✅ FaceMatrixLab 3D 渲染器初始化完成")
        
    def download_mediapipe_model(self):
        """下载MediaPipe人脸标志检测模型"""
        model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        model_path = "face_landmarker.task"
        
        if not os.path.exists(model_path):
            print("正在下载MediaPipe模型...")
            try:
                import urllib.request
                urllib.request.urlretrieve(model_url, model_path)
                print(f"✅ 模型下载完成: {model_path}")
            except Exception as e:
                print(f"❌ 模型下载失败: {e}")
                return None
        else:
            print(f"✅ MediaPipe模型已存在: {model_path}")
        
        return model_path
    
    def setup_camera_parameters(self):
        """设置相机参数（50mm等效焦距）"""
        # 50mm 等效焦距参数
        f_mm = 50.0  # 焦距(mm)
        sensor_width_mm = 36.0  # 全画幅传感器宽度(mm)
        
        # 计算像素焦距
        self.fx = (f_mm / sensor_width_mm) * self.render_width
        self.fy = (f_mm / sensor_width_mm) * self.render_height  # 假设正方形像素
        self.cx = self.render_width / 2.0
        self.cy = self.render_height / 2.0
        
        print(f"📷 相机参数设置:")
        print(f"   分辨率: {self.render_width}x{self.render_height}")
        print(f"   焦距: fx={self.fx:.2f}, fy={self.fy:.2f}")
        print(f"   主点: cx={self.cx:.2f}, cy={self.cy:.2f}")
        
        # 创建Open3D相机内参
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.render_width, self.render_height, 
            self.fx, self.fy, self.cx, self.cy
        )
    
    def load_face_model(self):
        """加载3D人脸模型"""
        if not os.path.exists(self.model_path):
            print(f"❌ 模型文件不存在: {self.model_path}")
            return False
            
        print(f"📦 正在加载3D模型: {self.model_path}")
        
        try:
            # 加载模型
            self.face_mesh = o3d.io.read_triangle_mesh(self.model_path)
            
            if len(self.face_mesh.vertices) == 0:
                print("❌ 模型加载失败：没有顶点数据")
                return False
            
            # 计算法线
            self.face_mesh.compute_vertex_normals()
            
            # 获取顶点信息
            vertices = np.asarray(self.face_mesh.vertices)
            self.num_vertices = len(vertices)
            
            print(f"✅ 模型加载成功:")
            print(f"   顶点数: {self.num_vertices}")
            print(f"   面数: {len(self.face_mesh.triangles)}")
            print(f"   坐标范围:")
            print(f"     X: [{vertices[:, 0].min():.2f}, {vertices[:, 0].max():.2f}] mm")
            print(f"     Y: [{vertices[:, 1].min():.2f}, {vertices[:, 1].max():.2f}] mm") 
            print(f"     Z: [{vertices[:, 2].min():.2f}, {vertices[:, 2].max():.2f}] mm")
            
            # 设置材质
            self.face_mesh.paint_uniform_color([0.8, 0.7, 0.6])  # 肤色
            
            # 备份原始顶点
            self.original_vertices = np.asarray(self.face_mesh.vertices).copy()
            
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def setup_visualizer(self):
        """设置Open3D可视化器（固定视角版本）"""
        print("🎨 初始化Open3D可视化器...")
        
        # 创建可视化器
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("FaceMatrixLab - 3D Face Renderer", self.render_width, self.render_height)
        
        # 添加模型到可视化器
        self.vis.add_geometry(self.face_mesh)
        
        # 设置渲染选项
        render_option = self.vis.get_render_option()
        render_option.mesh_show_back_face = True
        render_option.background_color = np.array([0.1, 0.1, 0.1])
        
        # 设置相机参数（固定视角）
        view_control = self.vis.get_view_control()
        
        print("✅ Open3D可视化器初始化完成")
        return True
    
    def create_mediapipe_landmarker(self):
        """创建MediaPipe人脸标志检测器"""
        if not self.mp_model_path:
            return None
            
        try:
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.mp_model_path),
                running_mode=VisionRunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,  # 暂时不用blendshapes
                output_facial_transformation_matrixes=True,  # 输出变换矩阵
            )
            
            landmarker = FaceLandmarker.create_from_options(options)
            print("✅ MediaPipe人脸检测器创建成功")
            return landmarker
            
        except Exception as e:
            print(f"❌ MediaPipe检测器创建失败: {e}")
            return None
    
    def detection_thread(self):
        """MediaPipe检测线程"""
        print("🎥 启动MediaPipe检测线程...")
        
        # 创建检测器
        self.landmarker = self.create_mediapipe_landmarker()
        if not self.landmarker:
            print("❌ MediaPipe检测器初始化失败")
            return
        
        # 打开摄像头
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"❌ 无法打开摄像头 {self.camera_id}")
            return
        
        # 设置摄像头分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        frame_count = 0
        fps_start_time = time.time()
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame = cv2.flip(frame, 1)  # 镜像翻转
                frame_count += 1
                
                # 转换为RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # 进行人脸检测
                timestamp_ms = int(time.time() * 1000)
                detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
                
                # 将结果放入队列
                try:
                    data_packet = {
                        'detection_result': detection_result,
                        'frame': rgb_frame,
                        'timestamp': timestamp_ms
                    }
                    self.data_queue.put_nowait(data_packet)
                except queue.Full:
                    # 如果队列满了，丢弃最旧的数据
                    try:
                        self.data_queue.get_nowait()
                        self.data_queue.put_nowait(data_packet)
                    except queue.Empty:
                        pass
                
                # 计算检测FPS
                if frame_count % 30 == 0:
                    elapsed = time.time() - fps_start_time
                    self.current_detection_fps = 30.0 / elapsed
                    fps_start_time = time.time()
                
                # 控制帧率
                time.sleep(1.0 / self.fps_target)
                
        except Exception as e:
            print(f"❌ 检测线程错误: {e}")
        finally:
            cap.release()
            print("🎥 MediaPipe检测线程已停止")
    
    def update_face_model(self, detection_result):
        """根据MediaPipe结果更新3D人脸模型"""
        if not detection_result.face_landmarks:
            return False
        
        # 获取变换矩阵
        if (detection_result.facial_transformation_matrixes and 
            len(detection_result.facial_transformation_matrixes) > 0):
            
            facial_transform = np.array(detection_result.facial_transformation_matrixes[0])
            
            # 应用变换到原始顶点
            vertices = self.original_vertices.copy()
            
            # 将顶点转换为齐次坐标
            vertices_homogeneous = np.hstack([vertices, np.ones((len(vertices), 1))])
            
            # 应用变换矩阵
            transformed_vertices = (facial_transform @ vertices_homogeneous.T).T[:, :3]
            
            # 更新模型顶点
            self.face_mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
            self.face_mesh.compute_vertex_normals()
            
            return True
        
        return False
    

    
    def run_with_visualizer(self):
        """使用Open3D可视化器运行（固定视角版本）"""
        print("🎬 启动Open3D可视化器...")
        
        # 设置可视化器
        if not self.setup_visualizer():
            print("❌ 可视化器初始化失败")
            return
        
        print("✅ 可视化器初始化完成")
        
        try:
            while self.is_running:
                # 处理MediaPipe数据
                try:
                    while not self.data_queue.empty():
                        data_packet = self.data_queue.get_nowait()
                        if self.update_face_model(data_packet['detection_result']):
                            # 更新可视化器中的几何体
                            self.vis.update_geometry(self.face_mesh)
                except queue.Empty:
                    pass
                
                # 更新可视化器
                if not self.vis.poll_events():
                    break
                self.vis.update_renderer()
                
                time.sleep(1.0 / 60)  # 60FPS渲染
                
        except KeyboardInterrupt:
            print("\n⏹️ 用户中断")
        finally:
            self.vis.destroy_window()
    
    def run(self):
        """启动渲染器"""
        print("\n🚀 启动FaceMatrixLab 3D渲染器（固定视角版本）")
        print("=" * 60)
        print("控制说明:")
        print("  固定视角显示")
        print("  Q键 或关闭窗口: 退出")
        print("=" * 60)
        
        self.is_running = True
        
        # 启动MediaPipe检测线程
        detection_thread = threading.Thread(target=self.detection_thread, daemon=True)
        detection_thread.start()
        
        # 等待检测线程启动
        time.sleep(2)
        
        try:
            # 运行可视化器
            self.run_with_visualizer()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("🧹 清理资源...")
        self.is_running = False
        print("✅ 资源清理完成")


def main():
    """主函数"""
    print("FaceMatrixLab 3D 人脸渲染器（固定视角版本）")
    print("使用MediaPipe + Open3D实现实时3D人脸追踪渲染")
    print("=" * 60)
    
    # 检查模型文件
    model_path = "../obj/Andy_Wah_facemesh.obj"
    if not os.path.exists(model_path):
        print(f"❌ 错误：模型文件不存在 {model_path}")
        print("请确保Andy_Wah_facemesh.obj文件位于 obj/ 目录中")
        return
    
    try:
        # 创建并运行渲染器
        renderer = FaceMatrixLabRenderer(camera_id=0, model_path=model_path)
        renderer.run()
        
    except Exception as e:
        print(f"❌ 程序运行错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("程序结束")


if __name__ == "__main__":
    main()
