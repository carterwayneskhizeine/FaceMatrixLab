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
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

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
        
        # 初始化Open3D渲染器
        if not self.setup_renderer():
            raise Exception("无法初始化Open3D渲染器")
        
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
            
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def setup_renderer(self):
        """设置Open3D渲染器"""
        print("🎨 初始化Open3D渲染器...")
        
        # 创建应用程序
        gui.Application.instance.initialize()
        
        # 创建窗口
        self.window = gui.Application.instance.create_window(
            "FaceMatrixLab - 3D Face Renderer", 
            self.render_width, self.render_height
        )
        
        # 创建3D场景
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene)
        
        # 加载3D模型前需要先检查模型文件
        if not hasattr(self, 'face_mesh'):
            print("❌ 人脸模型未加载，无法设置渲染器")
            return False
        
        # 添加人脸模型到场景
        material = rendering.MaterialRecord()
        material.shader = "defaultLit"
        material.base_color = [0.8, 0.7, 0.6, 1.0]  # 肤色 RGBA
        self.scene.scene.add_geometry("face_model", self.face_mesh, material)
        print("✅ 人脸模型已添加到场景")
        
        # 计算模型的边界框来设置相机
        bbox = self.face_mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_extent()
        
        # 设置相机位置和视角
        # 使用边界框来设置相机
        camera_distance = max(extent) * 2  # 相机距离模型中心的距离
        camera_pos = center + np.array([0, 0, camera_distance])
        
        # 使用正确的setup_camera参数格式
        self.scene.setup_camera(60.0, bbox, center)
        
        # 添加灯光
        self.scene.scene.set_lighting(rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, np.array([0, -1, -1]))
        
        # 设置背景
        self.scene.scene.set_background([0.1, 0.1, 0.1, 1.0])  # 深灰色背景
        
        print("✅ Open3D渲染器初始化完成")
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
            return
        
        # 获取第一个检测到的人脸数据
        landmarks = detection_result.face_landmarks[0]
        
        # 获取facial_transformation_matrix
        if (detection_result.facial_transformation_matrixes and 
            len(detection_result.facial_transformation_matrixes) > 0):
            
            # MediaPipe的变换矩阵（从canonical到相机坐标系）
            facial_transform = np.array(detection_result.facial_transformation_matrixes[0])
            
            # 应用变换矩阵到模型
            # 注意：Open3D使用列主序矩阵，MediaPipe可能使用行主序
            transform_matrix = facial_transform.T  # 转置以确保正确的矩阵顺序
            
            # 更新模型的变换
            self.scene.scene.set_geometry_transform("face_model", transform_matrix)
            
            # 【可选】使用landmarks更新模型顶点（如果需要更精细的变形）
            if len(landmarks) >= 468 and self.num_vertices >= 468:
                self.update_vertices_with_landmarks(landmarks, facial_transform)
    
    def update_vertices_with_landmarks(self, landmarks, facial_transform):
        """使用landmarks更新模型顶点（可选的高级功能）"""
        # 这里可以实现更精细的顶点变形
        # 由于Andy_Wah_facemesh.obj与canonical模型拓扑一致，可以直接映射前468个顶点
        
        try:
            # 获取当前顶点
            vertices = np.asarray(self.face_mesh.vertices).copy()
            
            # 如果模型顶点数正好是468，可以直接替换
            if len(vertices) == 468:
                # 将landmarks转换为毫米坐标
                for i, lm in enumerate(landmarks[:468]):
                    # landmarks是归一化坐标，需要转换为3D坐标
                    # 这里使用facial_transform进行转换
                    canonical_point = np.array([lm.x, lm.y, lm.z, 1.0])
                    world_point = facial_transform @ canonical_point
                    vertices[i] = world_point[:3]
                
                # 更新模型顶点
                self.face_mesh.vertices = o3d.utility.Vector3dVector(vertices)
                self.face_mesh.compute_vertex_normals()
                
                # 更新场景中的几何体
                material = rendering.MaterialRecord()
                material.shader = "defaultLit"
                material.base_color = [0.8, 0.7, 0.6, 1.0]
                self.scene.scene.remove_geometry("face_model")
                self.scene.scene.add_geometry("face_model", self.face_mesh, material)
            
        except Exception as e:
            print(f"⚠️ 顶点更新失败: {e}")
    
    def render_loop(self):
        """主渲染循环"""
        print("🎬 启动渲染循环...")
        
        render_fps_counter = 0
        render_fps_start = time.time()
        
        def update_callback():
            nonlocal render_fps_counter, render_fps_start
            
            if not self.is_running:
                return
            
            # 处理数据队列
            try:
                while not self.data_queue.empty():
                    data_packet = self.data_queue.get_nowait()
                    self.latest_result = data_packet['detection_result']
            except queue.Empty:
                pass
            
            # 更新3D模型
            if self.latest_result:
                self.update_face_model(self.latest_result)
            
            # 计算渲染FPS
            render_fps_counter += 1
            if render_fps_counter % 30 == 0:
                elapsed = time.time() - render_fps_start
                self.current_render_fps = 30.0 / elapsed
                render_fps_start = time.time()
                
                # 在窗口标题显示FPS信息
                title = (f"FaceMatrixLab - 3D Face Renderer | "
                        f"Render FPS: {self.current_render_fps:.1f} | "
                        f"Detection FPS: {self.current_detection_fps:.1f}")
                self.window.title = title
            
            # 继续调度下一帧更新
            if self.is_running:
                gui.Application.instance.post_to_main_thread(self.window, update_callback)
        
        # 启动第一次更新
        gui.Application.instance.post_to_main_thread(self.window, update_callback)
        
        # 运行GUI主循环
        gui.Application.instance.run()
    
    def run(self):
        """启动渲染器"""
        print("\n🚀 启动FaceMatrixLab 3D渲染器")
        print("=" * 50)
        print("控制说明:")
        print("  鼠标左键 + 拖拽: 旋转视角")
        print("  鼠标右键 + 拖拽: 缩放")
        print("  鼠标滚轮: 缩放")
        print("  ESC 或关闭窗口: 退出")
        print("=" * 50)
        
        self.is_running = True
        
        # 启动MediaPipe检测线程
        detection_thread = threading.Thread(target=self.detection_thread, daemon=True)
        detection_thread.start()
        
        # 等待一下确保检测线程启动
        time.sleep(1)
        
        try:
            # 启动渲染循环（这会阻塞主线程）
            self.render_loop()
        except KeyboardInterrupt:
            print("\n⏹️ 用户中断")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("🧹 清理资源...")
        self.is_running = False
        
        # 清理Open3D资源
        try:
            gui.Application.instance.quit()
        except:
            pass
        
        print("✅ 资源清理完成")


def main():
    """主函数"""
    print("FaceMatrixLab 3D 人脸渲染器")
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
