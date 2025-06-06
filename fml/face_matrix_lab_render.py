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
        self.latest_frame = None
        
        # AR背景控制
        self.show_camera_background = True  # 默认显示摄像机背景
        self.background_image = None
        self.latest_camera_frame = None  # 保存最新的摄像机帧
        
        # 【新增】相机校准参数加载
        self.use_real_calibration = True  # 是否使用真实校准参数
        self.calibration_intrinsic_path = "Camera-Calibration/output/intrinsic.txt"  # 内参文件路径
        self.calibration_extrinsic_path = "Camera-Calibration/output/extrinsic.txt"  # 外参文件路径
        
        # 相机参数（将根据真实校准或手动设置）
        self.camera_fx = None
        self.camera_fy = None
        self.camera_cx = None
        self.camera_cy = None
        self.camera_skew = 0.0  # 倾斜参数
        
        # 加载真实相机校准参数
        self.load_camera_calibration()
        
        # 相机参数（50mm 等效焦距）- 将在setup_camera_parameters中根据校准结果设置
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
    
    def load_camera_calibration(self):
        """加载真实的相机校准参数"""
        if not self.use_real_calibration:
            print("📷 未启用真实相机校准，将使用默认估计参数")
            return
        
        try:
            # 加载内参矩阵
            if os.path.exists(self.calibration_intrinsic_path):
                print(f"📷 正在加载相机内参: {self.calibration_intrinsic_path}")
                
                # 读取内参文件
                with open(self.calibration_intrinsic_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 解析内参矩阵 - 支持多种格式
                lines = content.strip().split('\n')
                matrix_lines = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('A=') and '[' in line and ']' in line:
                        # 清理方括号并提取数字
                        line = line.replace('[', '').replace(']', '')
                        matrix_lines.append(line)
                
                if len(matrix_lines) >= 3:
                    # 解析3x3内参矩阵
                    intrinsic_matrix = []
                    for line in matrix_lines[:3]:
                        # 分割数字（处理可能的科学计数法）
                        values = []
                        parts = line.split()
                        for part in parts:
                            try:
                                values.append(float(part))
                            except ValueError:
                                continue
                        if len(values) >= 3:
                            intrinsic_matrix.append(values[:3])
                    
                    if len(intrinsic_matrix) == 3:
                        # 提取相机参数
                        A = np.array(intrinsic_matrix)
                        self.camera_fx = A[0, 0]  # fx
                        self.camera_fy = A[1, 1]  # fy
                        self.camera_cx = A[0, 2]  # cx (主点x坐标)
                        self.camera_cy = A[1, 2]  # cy (主点y坐标)
                        self.camera_skew = A[0, 1]  # skew (倾斜参数)
                        
                        print("✅ 成功加载相机内参:")
                        print(f"   fx (x方向焦距): {self.camera_fx:.2f}")
                        print(f"   fy (y方向焦距): {self.camera_fy:.2f}")
                        print(f"   cx (主点x坐标): {self.camera_cx:.2f}")
                        print(f"   cy (主点y坐标): {self.camera_cy:.2f}")
                        print(f"   skew (倾斜参数): {self.camera_skew:.4f}")
                    else:
                        raise ValueError("无法解析内参矩阵格式")
                else:
                    raise ValueError("内参文件格式不正确")
                    
            else:
                print(f"❌ 内参文件不存在: {self.calibration_intrinsic_path}")
                self.use_real_calibration = False
                return
            
            # 加载外参矩阵（可选，用于更复杂的3D投影）
            if os.path.exists(self.calibration_extrinsic_path):
                print(f"📷 检测到外参文件: {self.calibration_extrinsic_path}")
                # 注意：当前代码主要使用内参进行透视投影，外参暂不使用
                # 如果需要更精确的3D几何计算，可以在这里加载外参矩阵
            
            print("✅ 相机校准参数加载完成")
            
        except Exception as e:
            print(f"❌ 加载相机校准参数失败: {e}")
            print("⚠️ 将回退到手动估计相机参数")
            self.use_real_calibration = False
            # 重置相机参数
            self.camera_fx = None
            self.camera_fy = None
            self.camera_cx = None
            self.camera_cy = None
            self.camera_skew = 0.0
        
    def setup_camera_parameters(self):
        """设置相机参数（优先使用真实校准参数，否则使用50mm等效焦距估计）"""
        print("📷 相机参数设置:")
        print(f"   分辨率: {self.render_width}x{self.render_height}")
        
        # 如果成功加载了真实校准参数，直接使用
        if (self.use_real_calibration and 
            self.camera_fx is not None and self.camera_fy is not None and 
            self.camera_cx is not None and self.camera_cy is not None):
            
            self.fx = self.camera_fx
            self.fy = self.camera_fy
            self.cx = self.camera_cx
            self.cy = self.camera_cy
            
            print("✅ 使用真实相机校准参数:")
            print(f"   焦距: fx={self.fx:.2f}, fy={self.fy:.2f}")
            print(f"   主点: cx={self.cx:.2f}, cy={self.cy:.2f}")
            if self.camera_skew != 0.0:
                print(f"   倾斜: skew={self.camera_skew:.4f}")
        
        else:
            # 回退到50mm等效焦距估计
            print("⚠️ 使用50mm等效焦距估计参数:")
            
            # 50mm 等效焦距参数
            f_mm = 50.0  # 焦距(mm)
            sensor_width_mm = 36.0  # 全画幅传感器宽度(mm)
            
            # 计算像素焦距
            self.fx = (f_mm / sensor_width_mm) * self.render_width
            self.fy = (f_mm / sensor_width_mm) * self.render_height  # 假设正方形像素
            self.cx = self.render_width / 2.0
            self.cy = self.render_height / 2.0
            
            print(f"   焦距: fx={self.fx:.2f}, fy={self.fy:.2f}")
            print(f"   主点: cx={self.cx:.2f}, cy={self.cy:.2f}")
            print(f"   注意: 这是基于50mm等效焦距的估计值")
        
        # 创建Open3D相机内参
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.render_width, self.render_height, 
            self.fx, self.fy, self.cx, self.cy
        )
        
        print(f"✅ Open3D相机内参创建完成")
    
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
        """设置Open3D可视化器（离屏渲染版本，兼容Windows）"""
        print("🎨 初始化Open3D可视化器...")
        # 创建隐藏的可视化器窗口进行离屏渲染
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("_", self.render_width, self.render_height, visible=False)
        
        # 添加人脸模型（不使用MaterialRecord，直接使用网格颜色）
        self.vis.add_geometry(self.face_mesh)
        
        # 设置渲染选项
        render_option = self.vis.get_render_option()
        render_option.mesh_show_back_face = True
        render_option.background_color = np.array([0.0, 0.0, 0.0])  # 黑色背景便于合成
        
        # 设置相机参数
        ctr = self.vis.get_view_control()
        # 创建相机参数对象
        camera_params = o3d.camera.PinholeCameraParameters()
        camera_params.intrinsic = self.intrinsic
        camera_params.extrinsic = np.eye(4)
        ctr.convert_from_pinhole_camera_parameters(camera_params)
        
        print("✅ Open3D可视化器初始化完成")
        return True
    
    def setup_camera_background(self):
        """设置摄像机背景显示"""
        # 创建背景平面（用于显示摄像机画面）
        # 背景平面位于模型后方，大小匹配渲染视图
        background_vertices = np.array([
            [-200, -150, -300],  # 左下
            [200, -150, -300],   # 右下
            [200, 150, -300],    # 右上
            [-200, 150, -300]    # 左上
        ])
        
        background_triangles = np.array([
            [0, 1, 2],  # 第一个三角形
            [0, 2, 3]   # 第二个三角形
        ])
        
        # 创建背景网格
        self.background_mesh = o3d.geometry.TriangleMesh()
        self.background_mesh.vertices = o3d.utility.Vector3dVector(background_vertices)
        self.background_mesh.triangles = o3d.utility.Vector3iVector(background_triangles)
        self.background_mesh.compute_vertex_normals()
        
        # 设置UV坐标用于纹理映射
        uv_coordinates = np.array([
            [0, 0],  # 左下
            [1, 0],  # 右下
            [1, 1],  # 右上
            [0, 1]   # 左上
        ])
        
        # 初始时隐藏背景
        self.background_mesh.paint_uniform_color([0.1, 0.1, 0.1])  # 深灰色
        
        # 添加到可视化器
        self.vis.add_geometry(self.background_mesh)
        
        print("📺 摄像机背景平面已创建")
    
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
    
    def update_camera_background(self, frame):
        """更新摄像机背景显示"""
        if frame is None:
            return
            
        # 保存最新帧用于显示
        self.latest_camera_frame = frame.copy()
        
        try:
            if self.show_camera_background:
                # 方案1：在独立的OpenCV窗口中显示摄像机画面
                # 创建AR合成视图
                display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # 添加AR信息叠加
                overlay_text = "AR Background - Live Camera"
                cv2.putText(display_frame, overlay_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 显示摄像机背景窗口
                cv2.namedWindow("Camera Background", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Camera Background", 640, 480)
                cv2.imshow("Camera Background", display_frame)
                
                # 方案2：同时尝试改进3D背景平面的显示
                if hasattr(self, 'background_mesh'):
                    # 使用图像的整体亮度来调整背景平面
                    avg_brightness = np.mean(frame) / 255.0
                    
                    # 根据摄像机画面调整背景色
                    # 取不同区域的颜色
                    h, w = frame.shape[:2]
                    
                    # 更密集的采样以获得更好的效果
                    sample_points = [
                        (h//4, w//4),     # 左上
                        (h//4, 3*w//4),   # 右上
                        (3*h//4, 3*w//4), # 右下
                        (3*h//4, w//4)    # 左下
                    ]
                    
                    colors = []
                    for y, x in sample_points:
                        pixel_color = frame[y, x] / 255.0
                        # 增强颜色饱和度以便更好地显示
                        pixel_color = pixel_color * 1.5
                        pixel_color = np.clip(pixel_color, 0, 1)
                        colors.append(pixel_color)
                    
                    # 设置顶点颜色
                    self.background_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
                    
                                    # 更新可视化器中的背景
                self.vis.update_geometry(self.background_mesh)
                    
        except Exception as e:
            print(f"⚠️ 背景更新失败: {e}")
    
    def toggle_camera_background(self):
        """切换摄像机背景显示"""
        self.show_camera_background = not self.show_camera_background
        
        if self.show_camera_background:
            print("📺 摄像机背景已开启 - AR模式")
            # 显示最新的摄像机画面（如果有的话）
            if self.latest_camera_frame is not None:
                self.update_camera_background(self.latest_camera_frame)
        else:
            print("🎭 摄像机背景已关闭 - 纯3D模式") 
            # 关闭摄像机背景窗口
            try:
                cv2.destroyWindow("Camera Background")
            except:
                pass
            
            # 隐藏背景平面
            if hasattr(self, 'background_mesh'):
                self.background_mesh.paint_uniform_color([0.1, 0.1, 0.1])  # 深灰色
                self.vis.update_geometry(self.background_mesh)

    def run_with_visualizer(self):
        """使用Open3D隐藏窗口渲染并合成AR视图"""
        if not self.setup_visualizer():
            print("❌ 可视化器初始化失败")
            return
            
        cv2.namedWindow("AR View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AR View", self.render_width, self.render_height)
        
        try:
            while self.is_running:
                # 读取最新数据
                frame = None
                try:
                    pkt = self.data_queue.get_nowait()
                    detection_result = pkt['detection_result']
                    frame = pkt.get('frame')
                except queue.Empty:
                    detection_result = None
                
                # 更新3D模型
                if detection_result and self.update_face_model(detection_result):
                    self.vis.update_geometry(self.face_mesh)
                
                # 离屏渲染获取图像
                self.vis.poll_events()
                self.vis.update_renderer()
                img_3d = np.asarray(self.vis.capture_screen_float_buffer(False))
                img_3d = (img_3d * 255).astype(np.uint8)
                img_3d_bgr = cv2.cvtColor(img_3d, cv2.COLOR_RGB2BGR)
                
                # 摄像机背景
                if self.show_camera_background and frame is not None:
                    bg = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    bg = cv2.resize(bg, (self.render_width, self.render_height))
                else:
                    bg = np.zeros_like(img_3d_bgr)
                
                # AR合成：将3D模型叠加到背景上
                # 创建掩码：非黑色像素的区域
                mask = img_3d_bgr.sum(axis=2) > 30
                composite = bg.copy()
                composite[mask] = img_3d_bgr[mask]
                
                # 显示合成结果
                cv2.imshow("AR View", composite)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('o'):
                    self.toggle_camera_background()
                elif key == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n⏹️ 用户中断")
        finally:
            self.vis.destroy_window()
            cv2.destroyAllWindows()
    
    def run(self):
        """启动渲染器"""
        print("\n启动FaceMatrixLab 3D渲染器（AR增强现实版本）")
        print("=" * 60)
        print("📷 相机系统:")
        if self.use_real_calibration and self.camera_fx is not None:
            print("  ✅ 使用真实相机校准参数")
            print(f"  📂 内参文件: {self.calibration_intrinsic_path}")
        else:
            print("  ⚠️ 使用估计相机参数（50mm等效焦距）")
            print("  💡 如需精确渲染，请将相机校准文件放置在:")
            print(f"     {self.calibration_intrinsic_path}")
        print("=" * 60)
        print("控制说明:")
        print("  O键: 切换摄像机背景显示（AR模式 / 纯3D模式）")
        print("  Q键: 退出程序")
        print("  固定视角显示，3D模型叠加在真实摄像机画面上")
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
    print("FaceMatrixLab 3D 人脸渲染器（AR增强现实版本）")
    print("使用MediaPipe + Open3D实现实时3D人脸追踪渲染 + AR叠加效果")
    print("支持真实相机校准参数，提供更精确的3D投影效果")
    print("=" * 60)
    
    # 检查模型文件
    model_path = "obj/Andy_Wah_facemesh.obj"
    if not os.path.exists(model_path):
        print(f"❌ 错误：模型文件不存在 {model_path}")
        print("请确保Andy_Wah_facemesh.obj文件位于 obj/ 目录中")
        return
    
    # 检查相机校准文件（可选）
    calibration_path = "Camera-Calibration/output/intrinsic.txt"
    if os.path.exists(calibration_path):
        print(f"✅ 发现相机校准文件：{calibration_path}")
        print("将使用真实相机参数进行精确3D渲染")
    else:
        print(f"⚠️ 未发现相机校准文件：{calibration_path}")
        print("将使用默认估计参数（50mm等效焦距）")
        print("💡 如需获得最佳渲染效果，建议先进行相机校准")
    
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
