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
    def __init__(self, camera_id=0, model_path="obj/Andy_Wah_facemesh.obj"):
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
        
                # 【solvePnP方法】相机校准参数
        self.use_solvepnp = True  # 使用solvePnP进行精确3D跟踪
        self.calibration_file = "calib.npz"  # 相机标定文件
        
        # 相机内参矩阵
        self.K = None  # 3x3相机内参矩阵
        self.dist = None  # 畸变系数
        
        # 【solvePnP关键点配置】基于您提供的landmark分析
        # 基于canonical_face_model.obj分析的准确配置
        # 中线点索引（这些点的X坐标应该为0）
        self.centerline_indices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 94, 151, 152, 164, 168, 175, 195, 197, 199, 200]
       
        # 对称点对索引（左点索引, 右点索引）
        self.symmetric_pairs = [
            (3, 248), (7, 249), (20, 250), (21, 251), (22, 252), (23, 253), (24, 254), (25, 255),
            (26, 256), (27, 257), (28, 258), (29, 259), (30, 260), (31, 261), (32, 262), (33, 263),
            (34, 264), (35, 265), (36, 266), (37, 267), (38, 268), (39, 269), (40, 270), (41, 271),
            (42, 272), (43, 273), (44, 274), (45, 275), (46, 276), (47, 277), (48, 278), (49, 279),
            (50, 280), (51, 281), (52, 282), (53, 283), (54, 284), (55, 285), (56, 286), (57, 287),
            (58, 288), (59, 289), (60, 290), (61, 291), (62, 292), (63, 293), (64, 294), (65, 295),
            (66, 296), (67, 297), (68, 298), (69, 299), (70, 300), (71, 301), (72, 302), (73, 303)
        ]
       
        # 关键landmark点
        self.key_landmarks = {
            'nose_tip': 4,      # 鼻尖
            'left_eye': 34,     # 左眼角
            'right_eye': 264,   # 右眼角
            'left_mouth': 192,  # 左嘴角
            'right_mouth': 416  # 右嘴角
        }
        
        # 【solvePnP用的3D-2D对应点】基于landmark分析选择稳定的关键点
        # 选择MediaPipe最稳定、最准确的检测点
        self.pnp_indices = [
            # 核心面部特征点（最稳定）
            1,     # 鼻尖中心
            168,   # 面部中心点
            10,    # 上唇中心
            152,   # 眉心中心
            175,   # 下唇中心
            
            # 双眼关键点（高精度）
            33, 263,   # 双眼内角
            130, 359,  # 双眼外角
            
            # 眉毛和眼部轮廓
            70, 300,   # 眉毛中部
            107, 336,  # 眼部轮廓
            
            # 鼻部特征
            19, 94,    # 鼻桥
            
            # 面部边界点
            234, 454,  # 面颊
            172, 397,  # 面部侧边
        ]
        
        # AR跟踪控制参数
        self.ar_tracking_enabled = True    # 是否启用AR跟踪
        self.coordinate_system_flip_z = False  # 是否翻转Z轴坐标
        self.ar_scale_factor = 1.0         # AR模型缩放系数
        self.ar_offset_x = 0.0             # AR模型X轴偏移
        self.ar_offset_y = 0.0             # AR模型Y轴偏移 
        self.ar_offset_z = 0.0             # AR模型Z轴偏移
        
        # 【调试模式】
        self.fallback_to_matrix = False    # 如果solvePnP失败，是否回退到transformation_matrix方法
        self.debug_mode = True             # 是否输出详细调试信息
        
        # 【solvePnP方法】加载相机标定参数
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
        """【solvePnP方法】加载相机标定参数"""
        if not self.use_solvepnp:
            print("📷 未启用solvePnP，将使用默认估计参数")
            return
        
        try:
            if os.path.exists(self.calibration_file):
                print(f"📷 正在加载相机标定文件: {self.calibration_file}")
                
                # 加载numpy标定文件
                calib_data = np.load(self.calibration_file)
                self.K = calib_data['K']  # 3x3相机内参矩阵
                self.dist = calib_data['dist']  # 畸变系数
                
                print("✅ 相机标定参数加载成功:")
                print("📏 相机内参矩阵 (K):")
                print(self.K)
                print(f"   焦距: fx={self.K[0,0]:.2f}, fy={self.K[1,1]:.2f}")
                print(f"   主点: cx={self.K[0,2]:.2f}, cy={self.K[1,2]:.2f}")
                print(f"🔧 畸变系数: {self.dist.ravel()}")
                
                # 检查是否有质量信息
                if 'mean_error' in calib_data:
                    mean_error = calib_data['mean_error']
                    print(f"📊 标定精度: {mean_error:.3f} 像素")
                    
                print("🎯 solvePnP方法已启用，将进行精确3D姿态估计")
                
            else:
                print(f"❌ 标定文件不存在: {self.calibration_file}")
                print("🔄 尝试加载兼容格式的标定文件...")
                
                # 尝试加载face_landmarker_cmaera_new.py格式的标定文件
                intrinsic_path = "Camera-Calibration/output/intrinsic.txt"
                if self.load_calibration_from_text(intrinsic_path):
                    print("✅ 成功从文本格式加载相机标定")
                else:
                    print("请先运行 python calibrate_cam.py 进行相机标定")
                    self.use_solvepnp = False
                
        except Exception as e:
            print(f"❌ 加载相机标定参数失败: {e}")
            print("⚠️ 将回退到估计相机参数")
            self.use_solvepnp = False
            self.K = None
            self.dist = None
    
    def load_calibration_from_text(self, intrinsic_path):
        """从文本格式加载相机标定参数（兼容face_landmarker_cmaera_new.py）"""
        try:
            if os.path.exists(intrinsic_path):
                print(f"📄 正在加载文本格式标定文件: {intrinsic_path}")
                
                # 读取内参文件
                with open(intrinsic_path, 'r', encoding='utf-8') as f:
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
                        self.K = np.array(intrinsic_matrix)
                        
                        # 设置默认畸变系数（如果没有专门的畸变文件）
                        self.dist = np.zeros((5,), dtype=np.float64)
                        
                        print("✅ 从文本格式加载相机内参:")
                        print(f"   fx: {self.K[0,0]:.2f}")
                        print(f"   fy: {self.K[1,1]:.2f}")
                        print(f"   cx: {self.K[0,2]:.2f}")
                        print(f"   cy: {self.K[1,2]:.2f}")
                        print(f"   畸变系数: 使用零畸变（可按需调整）")
                        
                        self.use_solvepnp = True
                        return True
                    else:
                        print("❌ 无法解析内参矩阵格式")
                        return False
                else:
                    print("❌ 内参文件格式不正确")
                    return False
            else:
                print(f"❌ 文本格式标定文件不存在: {intrinsic_path}")
                return False
                
        except Exception as e:
            print(f"❌ 加载文本格式标定失败: {e}")
            return False
    
    def setup_camera_parameters(self):
        """【solvePnP方法】设置相机参数（优先使用标定参数，否则使用估计值）"""
        print("📷 相机参数设置:")
        print(f"   分辨率: {self.render_width}x{self.render_height}")
        
        # 如果成功加载了标定参数，直接使用
        if self.use_solvepnp and self.K is not None:
            self.fx = self.K[0, 0]
            self.fy = self.K[1, 1]
            self.cx = self.K[0, 2]
            self.cy = self.K[1, 2]
            
            print("✅ 使用标定相机参数:")
            print(f"   焦距: fx={self.fx:.2f}, fy={self.fy:.2f}")
            print(f"   主点: cx={self.cx:.2f}, cy={self.cy:.2f}")
            print("   🎯 solvePnP将使用这些精确参数进行3D姿态估计")
        
        else:
            # 回退到50mm等效焦距估计
            print("⚠️ 使用50mm等效焦距估计参数:")
            
        # 50mm 等效焦距参数
        f_mm = 50.0  # 焦距(mm)
        sensor_width_mm = 36.0  # 全画幅传感器宽度(mm)
        
        # 计算像素焦距
        self.fx = (f_mm / sensor_width_mm) * self.render_width
        self.fy = (f_mm / sensor_width_mm) * self.render_height
        self.cx = self.render_width / 2.0
        self.cy = self.render_height / 2.0
        
        # 创建估计的K矩阵和畸变系数（用于solvePnP）
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist = np.zeros(5, dtype=np.float32)  # 假设无畸变
        
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
        """加载3D人脸模型（使用与测试脚本相同的方式）"""
        if not os.path.exists(self.model_path):
            print(f"❌ 模型文件不存在: {self.model_path}")
            return False
            
        print(f"📦 正在加载3D模型: {self.model_path}")
        
        try:
            # 【关键修正】直接从OBJ文件读取前468个顶点，与MediaPipe landmarks匹配
            vertices = []
            with open(self.model_path, 'r') as f:
                for line in f:
                    if line.startswith('v '):
                        parts = line.strip().split()
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        vertices.append([x, y, z])
                        # 只读取前468个顶点，匹配MediaPipe landmarks数量
                        if len(vertices) >= 468:
                            break
            
            vertices = np.array(vertices, dtype=np.float32)
            
            if len(vertices) == 0:
                print("❌ 模型加载失败：没有顶点数据")
                return False
            
            # 使用简化的顶点创建基础几何体
            # 创建点云用于可视化
            self.face_mesh = o3d.geometry.PointCloud()
            self.face_mesh.points = o3d.utility.Vector3dVector(vertices)
            self.face_mesh.paint_uniform_color([0.8, 0.7, 0.6])  # 肤色
            
            # 为了更好的可视化效果，也创建一个简单的三角网格
            # 读取完整的OBJ文件用于渲染
            self.face_mesh_full = o3d.io.read_triangle_mesh(self.model_path)
            if len(self.face_mesh_full.vertices) > 0:
                self.face_mesh_full.compute_vertex_normals()
                self.face_mesh_full.paint_uniform_color([0.8, 0.7, 0.6])
                # 使用完整网格进行显示
                self.face_mesh = self.face_mesh_full
            
            self.num_vertices = len(vertices)  # 用于solvePnP的顶点数量（468）
            
            print(f"✅ 模型加载成功:")
            print(f"   solvePnP顶点数: {self.num_vertices} (前468个)")
            print(f"   显示网格顶点数: {len(self.face_mesh.vertices)}")
            print(f"   面数: {len(self.face_mesh.triangles) if hasattr(self.face_mesh, 'triangles') else 0}")
            print(f"   坐标范围:")
            print(f"     X: [{vertices[:, 0].min():.2f}, {vertices[:, 0].max():.2f}] mm")
            print(f"     Y: [{vertices[:, 1].min():.2f}, {vertices[:, 1].max():.2f}] mm") 
            print(f"     Z: [{vertices[:, 2].min():.2f}, {vertices[:, 2].max():.2f}] mm")
            
            # 【重要】备份前468个顶点用于solvePnP
            self.original_vertices = vertices.copy()
            
            # 【关键修正】获取完整模型的原始顶点作为备份
            if hasattr(self, 'face_mesh_full') and len(self.face_mesh_full.vertices) > 0:
                self.original_full_vertices = np.asarray(self.face_mesh_full.vertices).copy()
            else:
                self.original_full_vertices = vertices.copy()
            
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
        
        # 调整near/far裁剪面，支持毫米级深度范围
        try:
            ctr.set_constant_z_near(1.0)
            ctr.set_constant_z_far(10000.0)
            print("🔧 视锥裁剪范围: near=1mm, far=10000mm")
        except AttributeError:
            # 某些Open3D版本可能不支持此方法
            pass
        
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
        """【solvePnP方法】根据MediaPipe结果更新3D人脸模型"""
        if not detection_result.face_landmarks:
            return False
        
        # 获取第一个检测到的人脸landmarks
        landmarks = detection_result.face_landmarks[0]
        
        # 【solvePnP核心】准备3D-2D对应点
        # 1. 2D图像点：从MediaPipe landmarks提取像素坐标
        img_points = []
        obj_points = []
        
        for idx in self.pnp_indices:
            if idx < len(landmarks) and idx < len(self.original_vertices):
                # MediaPipe输出的坐标是[0,1]范围，需要转换为像素坐标
                x_pixel = landmarks[idx].x * self.render_width
                y_pixel = landmarks[idx].y * self.render_height
                img_points.append([x_pixel, y_pixel])
                
                # 对应的3D模型点（Andy_Wah_facemesh.obj中的顶点）
                obj_points.append(self.original_vertices[idx])
        
        img_points = np.array(img_points, dtype=np.float32)
        obj_points = np.array(obj_points, dtype=np.float32)
        
        # 检查点数量是否足够
        if len(img_points) < 8 or len(obj_points) < 8:
            print(f"⚠️ solvePnP需要至少8个对应点，当前只有{len(img_points)}个")
            return False
        
        # 【关键改进】对3D点进行适当的缩放和调整
        # Andy_Wah_facemesh.obj的坐标可能需要缩放到合适的真实世界尺寸
        # 一般人脸宽度约15-18cm，我们需要确保3D模型尺寸合理
        
        # 计算模型的尺寸并调整到真实人脸尺寸
        model_bbox = np.array([
            [obj_points[:, 0].min(), obj_points[:, 1].min(), obj_points[:, 2].min()],
            [obj_points[:, 0].max(), obj_points[:, 1].max(), obj_points[:, 2].max()]
        ])
        model_width = model_bbox[1, 0] - model_bbox[0, 0]  # X方向宽度
        model_height = model_bbox[1, 1] - model_bbox[0, 1]  # Y方向高度
        
        # 【重要修正】根据Blender调试结果，减小缩放系数
        # 之前160mm导致11.433x缩放太大，用户需要0.1倍才能匹配
        # 改为更小的目标尺寸，让模型接近原始大小
        target_width = 14.0  # 毫米 (约为原始模型大小，避免过度缩放)
        scale_factor = target_width / model_width if model_width > 0 else 1.0
        
        # 缩放3D点到合理的真实世界尺寸
        obj_points_scaled = obj_points * scale_factor
        
        # 【调试】打印前几个对应点的坐标（初次运行时）
        if not hasattr(self, '_debug_points_printed'):
            self._debug_points_printed = True
            print("🔍 solvePnP对应点检查:")
            print(f"   模型原始尺寸: 宽{model_width:.2f} 高{model_height:.2f}")
            print(f"   缩放系数: {scale_factor:.3f} (目标宽度{target_width}mm)")
            for i in range(min(5, len(img_points))):
                print(f"   点{self.pnp_indices[i]}: 3D{obj_points_scaled[i]} -> 2D{img_points[i]}")
        
        # 【solvePnP核心】求解3D姿态（旋转和平移）
        try:
            # 使用SOLVEPNP_ITERATIVE算法，通常更稳定
            success, rvec, tvec = cv2.solvePnP(
                obj_points_scaled,  # 3D物体点（缩放后的毫米）
                img_points,         # 2D图像点（像素）
                self.K,             # 相机内参矩阵
                self.dist,          # 畸变系数
                flags=cv2.SOLVEPNP_ITERATIVE  # 使用迭代算法
            )
            
            if not success:
                print("❌ solvePnP求解失败")
                return False
                
        except Exception as e:
            print(f"❌ solvePnP异常: {e}")
            return False
        
        # 【转换结果】将Rodrigues向量转换为旋转矩阵
        R, _ = cv2.Rodrigues(rvec)
        T = tvec.reshape(3)
        
        # 【重要修正】坐标系转换
        # OpenCV/solvePnP坐标系: X向右，Y向下，Z向前（远离相机）
        # Open3D渲染坐标系: X向右，Y向上，Z向外（朝向用户）
        # 需要进行Y轴和Z轴翻转
        
        # 修正旋转矩阵（Y轴和Z轴翻转）
        flip_matrix = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ], dtype=np.float32)
        
        R_corrected = flip_matrix @ R
        T_corrected = flip_matrix @ T
        
        # 【应用变换】将旋转和平移应用到所有顶点
        # 使用完整模型的原始顶点进行变换
        if hasattr(self, 'original_full_vertices'):
            # 使用完整模型的原始顶点进行变换
            full_vertices_scaled = self.original_full_vertices * scale_factor
            # 应用旋转和平移：R @ vertices.T + T
            transformed_vertices = (R_corrected @ full_vertices_scaled.T).T + T_corrected
        else:
            # 回退到使用前468个顶点
            all_vertices_scaled = self.original_vertices * scale_factor
            transformed_vertices = (R_corrected @ all_vertices_scaled.T).T + T_corrected
        
        # 【用户控制参数】应用微调
        final_vertices = transformed_vertices.copy()
        
        if self.coordinate_system_flip_z:
            # 如果用户需要额外翻转Z轴
            final_vertices[:, 2] = -final_vertices[:, 2]
        
        # 应用缩放
        if self.ar_scale_factor != 1.0:
            center = np.mean(final_vertices, axis=0)
            final_vertices = center + (final_vertices - center) * self.ar_scale_factor
        
        # 应用偏移
        final_vertices[:, 0] += self.ar_offset_x
        final_vertices[:, 1] += self.ar_offset_y
        final_vertices[:, 2] += self.ar_offset_z
        
        # 保持当前单位（毫米），与原始模型保持一致
        self.face_mesh.vertices = o3d.utility.Vector3dVector(final_vertices)
        self.face_mesh.compute_vertex_normals()
        
        # 【调试信息】每10帧输出一次
        if not hasattr(self, '_solvepnp_frame_count'):
            self._solvepnp_frame_count = 0
        self._solvepnp_frame_count += 1
        
        if self.debug_mode and self._solvepnp_frame_count % 10 == 0:
            print(f"🎯 solvePnP结果 (第{self._solvepnp_frame_count}帧):")
            print(f"   原始平移: T=[{T[0]:.1f}, {T[1]:.1f}, {T[2]:.1f}] mm")
            print(f"   修正平移: T_corrected=[{T_corrected[0]:.1f}, {T_corrected[1]:.1f}, {T_corrected[2]:.1f}] mm")
            print(f"   旋转向量: rvec={rvec.ravel()}")
            print(f"   模型缩放: {scale_factor:.3f}x (目标真实尺寸)")
            
            center = np.mean(final_vertices, axis=0)
            z_range = (final_vertices[:, 2].min(), final_vertices[:, 2].max())
            print(f"   最终模型中心: [{center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}]")
            print(f"   最终Z深度范围: [{z_range[0]:.1f}, {z_range[1]:.1f}]")
            
            # 计算重投影误差（质量检查）
            try:
                reproj_points, _ = cv2.projectPoints(obj_points_scaled, rvec, tvec, self.K, self.dist)
                reproj_points = reproj_points.reshape(-1, 2)
                reproj_error = np.mean(np.linalg.norm(img_points - reproj_points, axis=1))
                print(f"   重投影误差: {reproj_error:.2f} 像素")
                
                # 优化重投影误差评判标准
                if reproj_error > 20:
                    print("⚠️ 重投影误差偏大，可能需要调整模型或标定")
                elif reproj_error < 10:
                    print("✅ 重投影误差良好")
                    
            except Exception as e:
                print(f"⚠️ 重投影误差计算失败: {e}")
        
        return True
    
    def update_face_model_fallback(self, detection_result):
        """【回退方法】使用MediaPipe的facial_transformation_matrix（用于调试对比）"""
        if not detection_result.face_landmarks:
            return False
        
        # 使用MediaPipe的facial_transformation_matrix进行AR跟踪
        if (detection_result.facial_transformation_matrixes and 
            len(detection_result.facial_transformation_matrixes) > 0):
            
            # 获取4×4面部变换矩阵
            facial_transform_matrix = np.array(detection_result.facial_transformation_matrixes[0])
            
            if self.debug_mode:
                print(f"🔄 回退方法 - 面部变换矩阵:")
                print(f"   平移: [{facial_transform_matrix[0,3]:.2f}, {facial_transform_matrix[1,3]:.2f}, {facial_transform_matrix[2,3]:.2f}]")
            
            # 获取原始模型顶点
            original_vertices = self.original_vertices.copy()
            
            # 将顶点转换为齐次坐标
            num_vertices = len(original_vertices)
            vertices_homogeneous = np.hstack([original_vertices, np.ones((num_vertices, 1))])
            
            # 应用变换矩阵
            transformed_vertices_homogeneous = (facial_transform_matrix @ vertices_homogeneous.T).T
            transformed_vertices = transformed_vertices_homogeneous[:, :3]
            
            # 应用用户控制参数
            final_vertices = transformed_vertices.copy()
            
            if self.coordinate_system_flip_z:
                final_vertices[:, 2] = -final_vertices[:, 2]
            
            if self.ar_scale_factor != 1.0:
                center = np.mean(final_vertices, axis=0)
                final_vertices = center + (final_vertices - center) * self.ar_scale_factor
            
            final_vertices[:, 0] += self.ar_offset_x
            final_vertices[:, 1] += self.ar_offset_y
            final_vertices[:, 2] += self.ar_offset_z
            
            # 保持当前单位（毫米），与原始模型保持一致
            self.face_mesh.vertices = o3d.utility.Vector3dVector(final_vertices)
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
    
    def export_current_model(self):
        """导出当前变换后的3D模型到OBJ文件"""
        try:
            if not hasattr(self.face_mesh, 'vertices') or len(self.face_mesh.vertices) == 0:
                print("❌ 无当前3D模型可导出")
                return False
            
            # 获取当前变换后的顶点
            current_vertices = np.asarray(self.face_mesh.vertices)
            
            # 生成文件名（带时间戳）
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output_realtime_model_{timestamp}.obj"
            
            # 确保output文件夹存在
            output_dir = "output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            filepath = os.path.join(output_dir, filename)
            
            # 写入OBJ文件
            with open(filepath, 'w') as f:
                # 写入文件头信息
                f.write(f"# FaceMatrixLab 实时3D人脸模型导出\n")
                f.write(f"# 导出时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# 基于模型: {self.model_path}\n")
                f.write(f"# 变换方法: {'solvePnP' if not self.fallback_to_matrix else 'MediaPipe transformation matrix'}\n")
                f.write(f"# 总顶点数: {len(current_vertices)}\n")
                f.write(f"# 模型参数:\n")
                f.write(f"#   缩放: {self.ar_scale_factor:.3f}x\n")
                f.write(f"#   偏移: X={self.ar_offset_x:.1f}, Y={self.ar_offset_y:.1f}, Z={self.ar_offset_z:.1f}\n")
                f.write(f"#   Z轴翻转: {'是' if self.coordinate_system_flip_z else '否'}\n")
                f.write("\n")
                
                # 写入顶点数据
                for i, vertex in enumerate(current_vertices):
                    f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
                
                # 如果原始模型有面信息，也写入面信息
                if hasattr(self.face_mesh, 'triangles') and len(self.face_mesh.triangles) > 0:
                    f.write("\n# 面信息\n")
                    triangles = np.asarray(self.face_mesh.triangles)
                    for triangle in triangles:
                        # OBJ文件的顶点索引从1开始
                        f.write(f"f {triangle[0]+1} {triangle[1]+1} {triangle[2]+1}\n")
            
            # 统计信息
            bbox_min = current_vertices.min(axis=0)
            bbox_max = current_vertices.max(axis=0)
            bbox_size = bbox_max - bbox_min
            
            print(f"✅ 实时3D模型已导出: {filepath}")
            print(f"📊 模型统计:")
            print(f"   顶点数: {len(current_vertices)}")
            print(f"   面数: {len(self.face_mesh.triangles) if hasattr(self.face_mesh, 'triangles') else 0}")
            print(f"   包围盒尺寸: {bbox_size[0]:.2f} x {bbox_size[1]:.2f} x {bbox_size[2]:.2f} mm")
            print(f"   中心位置: ({np.mean(current_vertices, axis=0)[0]:.2f}, {np.mean(current_vertices, axis=0)[1]:.2f}, {np.mean(current_vertices, axis=0)[2]:.2f}) mm")
            print(f"💡 在Blender中与原始模型 {self.model_path} 比较:")
            print(f"   1. 导入原始模型: {self.model_path}")
            print(f"   2. 导入实时模型: {filepath}")
            print(f"   3. 查看两个模型的相对位置和变换状态")
            
            return True
            
        except Exception as e:
            print(f"❌ 导出模型失败: {e}")
            import traceback
            traceback.print_exc()
            return False

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
                
                # 更新3D模型 - 可以在solvePnP和回退方法之间切换
                model_updated = False
                if detection_result:
                    if self.fallback_to_matrix:
                        # 使用回退方法（MediaPipe transformation matrix）
                        model_updated = self.update_face_model_fallback(detection_result)
                    else:
                        # 使用solvePnP方法
                        model_updated = self.update_face_model(detection_result)
                
                if model_updated:
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
                elif key == ord('q') or key == 27:  # Q键或ESC键：退出
                    break
                elif key == ord('f') or key == ord('F'):  # F键：切换Z轴翻转
                    self.coordinate_system_flip_z = not self.coordinate_system_flip_z
                    print(f"🔄 Z轴翻转: {'开启' if self.coordinate_system_flip_z else '关闭'}")
                elif key == ord('+') or key == ord('='):  # +键：放大AR模型
                    self.ar_scale_factor = min(5.0, self.ar_scale_factor + 0.1)
                    print(f"📏 AR模型缩放: {self.ar_scale_factor:.1f}x")
                elif key == ord('-') or key == ord('_'):  # -键：缩小AR模型
                    self.ar_scale_factor = max(0.1, self.ar_scale_factor - 0.1)
                    print(f"📏 AR模型缩放: {self.ar_scale_factor:.1f}x")
                elif key == 82:  # 上箭头：Y轴正向偏移
                    self.ar_offset_y -= 5.0
                    print(f"📍 AR模型Y偏移: {self.ar_offset_y:.1f}")
                elif key == 84:  # 下箭头：Y轴负向偏移
                    self.ar_offset_y += 5.0
                    print(f"📍 AR模型Y偏移: {self.ar_offset_y:.1f}")
                elif key == 81:  # 左箭头：X轴负向偏移
                    self.ar_offset_x -= 5.0
                    print(f"📍 AR模型X偏移: {self.ar_offset_x:.1f}")
                elif key == 83:  # 右箭头：X轴正向偏移
                    self.ar_offset_x += 5.0
                    print(f"📍 AR模型X偏移: {self.ar_offset_x:.1f}")
                elif key == 85:  # Page Up：Z轴前移
                    self.ar_offset_z -= 10.0
                    print(f"📍 AR模型Z偏移: {self.ar_offset_z:.1f}")
                elif key == 86:  # Page Down：Z轴后移
                    self.ar_offset_z += 10.0
                    print(f"📍 AR模型Z偏移: {self.ar_offset_z:.1f}")
                elif key == ord('m') or key == ord('M'):  # M键：切换跟踪方法
                    self.fallback_to_matrix = not self.fallback_to_matrix
                    method_name = "MediaPipe transformation matrix" if self.fallback_to_matrix else "solvePnP"
                    print(f"🔄 切换跟踪方法: {method_name}")
                elif key == ord('d') or key == ord('D'):  # D键：切换调试模式
                    self.debug_mode = not self.debug_mode
                    print(f"🐛 调试模式: {'开启' if self.debug_mode else '关闭'}")
                elif key == ord('e') or key == ord('E'):  # E键：导出当前3D模型
                    print("📤 导出当前实时3D模型...")
                    self.export_current_model()
                elif key == ord('r') or key == ord('R'):  # R键：重置所有参数
                    self.ar_scale_factor = 1.0
                    self.ar_offset_x = 0.0
                    self.ar_offset_y = 0.0
                    self.ar_offset_z = 0.0
                    self.coordinate_system_flip_z = False
                    print("🔄 所有AR参数已重置")
                elif key == ord('s') or key == ord('S'):  # S键：显示当前参数状态
                    print("📊 当前AR参数状态:")
                    print(f"   缩放: {self.ar_scale_factor:.2f}x")
                    print(f"   偏移: X={self.ar_offset_x:.1f}, Y={self.ar_offset_y:.1f}, Z={self.ar_offset_z:.1f}")
                    print(f"   Z轴翻转: {'是' if self.coordinate_system_flip_z else '否'}")
                    print(f"   跟踪方法: {'MediaPipe transformation matrix' if self.fallback_to_matrix else 'solvePnP'}")
                elif key == ord('['):  # [键：大幅缩小模型（0.1倍）
                    self.ar_scale_factor = max(0.01, self.ar_scale_factor * 0.1)
                    print(f"📏 模型大幅缩小: {self.ar_scale_factor:.3f}x")
                elif key == ord(']'):  # ]键：大幅放大模型（10倍）
                    self.ar_scale_factor = min(100.0, self.ar_scale_factor * 10.0)
                    print(f"📏 模型大幅放大: {self.ar_scale_factor:.3f}x")
                    
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
        if self.use_solvepnp and self.K is not None:
            print("  ✅ 使用solvePnP + 真实相机标定参数")
            print(f"  📂 标定文件: {self.calibration_file}")
            print("  🎯 将进行精确的3D姿态估计")
        else:
            print("  ⚠️ 使用估计相机参数（50mm等效焦距）")
            print("  💡 如需精确渲染，请运行: python calibrate_cam.py")
        print("=" * 60)
        print("控制说明:")
        print("  O键: 切换摄像机背景显示（AR模式 / 纯3D模式）")
        print("  Q键: 退出程序")
        print("  固定视角显示，3D模型叠加在真实摄像机画面上")
        print("  【AR跟踪控制】:")
        print("  M键: 切换跟踪方法（solvePnP ⇄ transformation matrix）")
        print("  D键: 切换调试模式（显示/隐藏详细信息）")
        print("  E键: 导出当前实时3D模型到output/文件夹（用于Blender调试）")
        print("  F键: 切换Z轴翻转（解决坐标系不匹配问题）")
        print("  【模型缩放控制】:")
        print("  +/-键: 细微调整缩放（±0.1倍）")
        print("  [/]键: 大幅调整缩放（÷10倍/×10倍）")
        print("  【位置控制】:")
        print("  方向键: 调整AR模型位置偏移")
        print("    ↑↓: Y轴偏移    ←→: X轴偏移")
        print("  PgUp/PgDn: Z轴偏移（前后移动）")
        print("  【实用功能】:")
        print("  R键: 重置所有AR参数到默认值")
        print("  S键: 显示当前AR参数状态")
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
    print("FaceMatrixLab 3D 人脸渲染器（solvePnP精确AR版本）")
    print("使用MediaPipe + solvePnP + Open3D实现精确3D人脸追踪渲染")
    print("通过相机标定和solvePnP算法，实现毫米级精度的AR叠加效果")
    print("=" * 60)
    
    # 检查模型文件
    model_path = "obj/Andy_Wah_facemesh.obj"
    if not os.path.exists(model_path):
        print(f"❌ 错误：模型文件不存在 {model_path}")
        print("请确保Andy_Wah_facemesh.obj文件位于 obj/ 目录中")
        return
    
    # 检查相机标定文件（重要）
    calibration_path = "calib.npz"
    if os.path.exists(calibration_path):
        print(f"✅ 发现相机标定文件：{calibration_path}")
        print("将使用solvePnP进行精确3D姿态估计")
        
        # 显示标定信息
        try:
            calib_data = np.load(calibration_path)
            if 'mean_error' in calib_data:
                print(f"📊 标定精度: {calib_data['mean_error']:.3f} 像素")
        except:
            pass
    else:
        print(f"⚠️ 未发现相机标定文件：{calibration_path}")
        print("将使用估计参数，精度有限")
        print("💡 强烈建议先运行: python calibrate_cam.py 进行相机标定")
    
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
