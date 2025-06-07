#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FaceMatrixLab 3D 面具渲染器
使用 MediaPipe 的 faceLandmarks (468个NormalizedLandmark点) 来挂载并渲染 Andy_Wah_facemesh.obj 模型
重点使用额头、左脸颊、下巴、右脸颊这4个特定点实现AR跟踪效果
🆕 新增：基于 BlendShapes 的表情驱动系统
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
import threading
import queue
import open3d as o3d

# MediaPipe 导入
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

class FaceMaskRenderer:
    def __init__(self, camera_id=0, model_path="obj/Andy_Wah_facemesh.obj"):
        """初始化3D面具渲染器"""
        print("=== FaceMatrixLab 3D 面具渲染器初始化 ===")
        
        # 基本参数
        self.camera_id = camera_id
        self.model_path = model_path
        self.is_running = False
        
        # 渲染参数
        self.render_width = 1280
        self.render_height = 720
        self.fps_target = 30
        
        # 🔑 重要：添加宽高比处理（参考face_landmarker_cmaera_new.py）
        self.camera_width = 1280   # 摄像头分辨率
        self.camera_height = 720   # 摄像头分辨率  
        self.aspect_ratio = self.camera_width / self.camera_height
        self.x_scale_factor = self.aspect_ratio / 1.0  # 对于16:9，约为1.777
        
        # MediaPipe 相关
        self.landmarker = None
        self.mp_model_path = self.download_mediapipe_model()
        
        # 🔑 关键：使用468个faceLandmarks中的4个特定点
        self.forehead_index = 10    # 额头
        self.left_cheek_index = 234  # 左脸颊
        self.chin_index = 152        # 下巴
        self.right_cheek_index = 454  # 右脸颊
        
        # 数据队列 - 用于线程间通信
        self.data_queue = queue.Queue(maxsize=5)
        
        # AR渲染相关
        self.show_camera_background = True  # 默认显示摄像机背景
        self.latest_camera_frame = None     # 保存最新的摄像机帧
        
        # 面具设置
        self.current_mask_color = [0.8, 0.7, 0.6]  # 默认肤色
        self.mask_colors = [
            [0.8, 0.7, 0.6],  # 肤色
            [0.9, 0.1, 0.1],  # 红色
            [0.1, 0.8, 0.2],  # 绿色
            [0.2, 0.2, 0.9],  # 蓝色
            [0.9, 0.7, 0.1],  # 金色
            [0.7, 0.3, 0.9],  # 紫色
        ]
        self.current_color_index = 0
        
        # 调试和跟踪
        self.debug_mode = True
        self.frame_count = 0
        
        # 🆕 新增：原始landmarks显示控制
        self.show_original_landmarks = True  # 显示原始landmarks点和线框
        
        # 🆕 新增：表情驱动系统
        self.enable_expression_drive = True  # 启用表情驱动
        self.show_blendshapes_debug = False  # 显示BlendShapes调试信息
        self.expression_strength = 1.0       # 表情强度系数
        
        # 🆕 新增：纹理贴图控制
        self.texture_mode = True             # 优先使用纹理贴图
        self.has_texture = False             # 是否成功加载纹理
        
        # 表情驱动相关的顶点组（需要根据具体模型调整）
        self.setup_expression_vertex_groups()
        
        # 加载3D模型
        if not self.load_face_model():
            raise Exception("无法加载3D模型文件")
        
        # 🆕 关键：提取模型中4个关键顶点的坐标
        self.extract_model_key_points()
        
        # 🆕 自动检测表情顶点组
        self.auto_detect_expression_vertex_groups()
        
        # 性能统计
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        print("✅ FaceMatrixLab 3D 面具渲染器初始化完成")
        print(f"📐 宽高比设置: {self.aspect_ratio:.3f} (16:9)")
        print(f"📏 X坐标修正系数: {self.x_scale_factor:.3f}")
        print(f"🎭 表情驱动: {'启用' if self.enable_expression_drive else '禁用'}")
    
    def setup_expression_vertex_groups(self):
        """🆕 设置表情驱动的顶点组"""
        print("🎭 设置表情驱动顶点组...")
        
        # 这里需要根据你的具体模型文件来定义哪些顶点属于嘴巴、眼睛等部位
        # 以下是示例索引，你需要根据实际模型调整
        
        # 嘴巴相关顶点（示例）
        self.mouth_vertex_indices = list(range(100, 150))  # 需要根据实际模型调整
        
        # 左眼相关顶点（示例）
        self.left_eye_vertex_indices = list(range(200, 230))  # 需要根据实际模型调整
        
        # 右眼相关顶点（示例）
        self.right_eye_vertex_indices = list(range(250, 280))  # 需要根据实际模型调整
        
        # 眉毛相关顶点（示例）
        self.eyebrow_vertex_indices = list(range(300, 350))  # 需要根据实际模型调整
        
        print(f"   嘴巴顶点: {len(self.mouth_vertex_indices)} 个")
        print(f"   左眼顶点: {len(self.left_eye_vertex_indices)} 个")
        print(f"   右眼顶点: {len(self.right_eye_vertex_indices)} 个")
        print(f"   眉毛顶点: {len(self.eyebrow_vertex_indices)} 个")
        
        # 💡 自动检测顶点组的方法（基于模型几何分析）
        # 这个方法会在load_face_model()之后调用
        
    def auto_detect_expression_vertex_groups(self):
        """🆕 自动检测表情相关的顶点组（基于几何位置）"""
        if not hasattr(self, 'original_vertices'):
            return
            
        vertices = self.original_vertices
        print("🔍 自动检测表情顶点组...")
        
        # 计算模型的边界框
        min_bounds = vertices.min(axis=0)
        max_bounds = vertices.max(axis=0)
        center = (min_bounds + max_bounds) / 2
        
        # 基于相对位置自动分组
        mouth_candidates = []
        left_eye_candidates = []
        right_eye_candidates = []
        eyebrow_candidates = []
        
        for i, vertex in enumerate(vertices):
            # 相对于中心的位置
            rel_pos = vertex - center
            
            # 嘴巴区域：下半部分，中央区域
            if (rel_pos[1] < -10 and  # Y轴负方向（下方）
                abs(rel_pos[0]) < 30 and  # X轴中央
                rel_pos[2] > -5):  # Z轴前方
                mouth_candidates.append(i)
            
            # 左眼区域：上方偏左
            elif (rel_pos[1] > 5 and  # Y轴正方向（上方）
                  rel_pos[0] < -15 and  # X轴负方向（左侧）
                  rel_pos[2] > -10):
                left_eye_candidates.append(i)
            
            # 右眼区域：上方偏右  
            elif (rel_pos[1] > 5 and  # Y轴正方向（上方）
                  rel_pos[0] > 15 and  # X轴正方向（右侧）
                  rel_pos[2] > -10):
                right_eye_candidates.append(i)
            
            # 眉毛区域：更上方
            elif (rel_pos[1] > 20 and  # Y轴更高
                  abs(rel_pos[0]) < 40 and
                  rel_pos[2] > -15):
                eyebrow_candidates.append(i)
        
        # 更新顶点组
        if mouth_candidates:
            self.mouth_vertex_indices = mouth_candidates
        if left_eye_candidates:
            self.left_eye_vertex_indices = left_eye_candidates
        if right_eye_candidates:
            self.right_eye_vertex_indices = right_eye_candidates
        if eyebrow_candidates:
            self.eyebrow_vertex_indices = eyebrow_candidates
        
        print(f"✅ 自动检测完成:")
        print(f"   嘴巴顶点: {len(self.mouth_vertex_indices)} 个")
        print(f"   左眼顶点: {len(self.left_eye_vertex_indices)} 个")
        print(f"   右眼顶点: {len(self.right_eye_vertex_indices)} 个")
        print(f"   眉毛顶点: {len(self.eyebrow_vertex_indices)} 个")
    
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
    
    def load_face_model(self):
        """加载3D人脸模型"""
        if not os.path.exists(self.model_path):
            print(f"❌ 模型文件不存在: {self.model_path}")
            return False
            
        print(f"📦 正在加载3D模型: {self.model_path}")
        
        try:
            # 加载模型（启用后处理以读取材质信息）
            self.face_mesh = o3d.io.read_triangle_mesh(self.model_path, enable_post_processing=True)
            
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
            
            # 🆕 新增：尝试加载纹理贴图
            texture_path = "obj/enhanced_texture.png"
            if os.path.exists(texture_path):
                try:
                    print(f"🎨 正在加载纹理贴图: {texture_path}")
                    tex_img = o3d.io.read_image(texture_path)
                    
                    # 🔧 修复UV坐标问题：上下翻转纹理
                    print("🔄 修正纹理方向（上下翻转）...")
                    img_array = np.asarray(tex_img)
                    
                    # 上下翻转图像数组
                    flipped_array = np.flipud(img_array)
                    
                    # 转换回Open3D图像格式
                    flipped_tex_img = o3d.geometry.Image(flipped_array)
                    self.face_mesh.textures = [flipped_tex_img]
                    
                    # 如果模型没有材质ID，设置所有面都使用纹理索引0
                    if len(self.face_mesh.triangle_material_ids) == 0:
                        self.face_mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(self.face_mesh.triangles))
                    
                    print(f"✅ 纹理贴图加载成功")
                    # 获取图像尺寸（Open3D方式）
                    height, width = img_array.shape[:2]
                    print(f"   纹理尺寸: {width} x {height}")
                    print(f"   已修正UV坐标方向")
                    self.has_texture = True
                    
                except Exception as e:
                    print(f"⚠️ 纹理加载失败: {e}")
                    self.has_texture = False
            else:
                print(f"⚠️ 纹理文件不存在: {texture_path}")
                self.has_texture = False
            
            # 🔧 修改：只有当没有纹理时才使用统一颜色
            if not hasattr(self, 'has_texture') or not self.has_texture:
                print("🎨 使用统一颜色渲染")
                self.change_mask_color(self.current_color_index)
            else:
                print("🎨 使用纹理贴图渲染")
            
            # 备份原始顶点
            self.original_vertices = np.asarray(self.face_mesh.vertices).copy()
            
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False

    def change_mask_color(self, color_index):
        """更改面具颜色"""
        if 0 <= color_index < len(self.mask_colors):
            self.current_color_index = color_index
            self.current_mask_color = self.mask_colors[color_index]
            
            # 🔧 修改：只有在没有纹理时才应用统一颜色
            if not hasattr(self, 'has_texture') or not self.has_texture:
                self.face_mesh.paint_uniform_color(self.current_mask_color)
                print(f"🎭 面具颜色已更改为索引 {color_index}")
            else:
                print(f"🎨 当前使用纹理贴图，颜色切换已禁用")
            return True
        return False
    
    def toggle_texture_mode(self):
        """🆕 切换纹理/颜色模式"""
        if self.has_texture:
            self.texture_mode = not self.texture_mode
            
            if self.texture_mode:
                print("🎨 切换到纹理贴图模式")
                # 清除顶点颜色，恢复纹理
                self.face_mesh.vertex_colors = o3d.utility.Vector3dVector([])
            else:
                print("🎨 切换到统一颜色模式")
                # 应用统一颜色
                self.face_mesh.paint_uniform_color(self.current_mask_color)
        else:
            print("⚠️ 没有可用的纹理贴图，无法切换")
    
    def setup_visualizer(self):
        """设置Open3D可视化器"""
        print("🎨 初始化Open3D可视化器...")
        # 创建可视化器窗口（离屏渲染）
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("FaceMask Renderer", self.render_width, self.render_height, visible=False)
        
        # 添加人脸模型
        self.vis.add_geometry(self.face_mesh)
        
        # 设置渲染选项
        render_option = self.vis.get_render_option()
        render_option.mesh_show_back_face = True
        render_option.background_color = np.array([0.0, 0.0, 0.0])  # 黑色背景便于合成
        
        # 设置相机视角
        ctr = self.vis.get_view_control()
        ctr.set_zoom(1.0)
        
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
                output_face_blendshapes=True,  # 🔑 关键：启用BlendShapes输出
                output_facial_transformation_matrixes=False,  # 🔑 关键：不使用transformation_matrix
            )
            
            landmarker = FaceLandmarker.create_from_options(options)
            print("✅ MediaPipe人脸检测器创建成功")
            print("🎭 BlendShapes输出已启用")
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
                    self.current_fps = 30.0 / elapsed
                    fps_start_time = time.time()
                
                # 控制帧率
                time.sleep(1.0 / self.fps_target)
                
        except Exception as e:
            print(f"❌ 检测线程错误: {e}")
        finally:
            cap.release()
            print("🎥 MediaPipe检测线程已停止")
    
    def extract_model_key_points(self):
        """🆕 提取模型中4个关键顶点的3D坐标"""
        try:
            # 获取模型顶点
            vertices = np.asarray(self.face_mesh.vertices)
            
            # 提取4个关键点（index与landmarks一致）
            self.model_forehead = vertices[self.forehead_index].copy()      # 额头: 10
            self.model_left_cheek = vertices[self.left_cheek_index].copy()  # 左脸颊: 234
            self.model_chin = vertices[self.chin_index].copy()              # 下巴: 152
            self.model_right_cheek = vertices[self.right_cheek_index].copy() # 右脸颊: 454
            
            print(f"🎯 模型关键点提取成功:")
            print(f"   额头[{self.forehead_index}]: {self.model_forehead}")
            print(f"   左脸颊[{self.left_cheek_index}]: {self.model_left_cheek}")
            print(f"   下巴[{self.chin_index}]: {self.model_chin}")
            print(f"   右脸颊[{self.right_cheek_index}]: {self.model_right_cheek}")
            
            # 计算模型的尺寸信息
            model_width = np.linalg.norm(self.model_right_cheek - self.model_left_cheek)
            model_height = np.linalg.norm(self.model_chin - self.model_forehead)
            print(f"   模型面部尺寸: 宽度={model_width:.2f}mm, 高度={model_height:.2f}mm")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型关键点提取失败: {e}")
            return False
    
    def update_face_model(self, detection_result):
        """🔑 关键：基于4个landmarks点的屏幕位置计算旋转缩放移动"""
        if not detection_result.face_landmarks or len(detection_result.face_landmarks) == 0:
            return False
        
        # 获取第一个检测到的人脸的468个关键点
        landmarks = detection_result.face_landmarks[0]
        
        # 确保有足够的关键点
        if len(landmarks) < 468:
            print(f"⚠️ 关键点数量不足: {len(landmarks)}, 期望468个")
            return False
        
        # 🔑 提取4个特定关键点 (NormalizedLandmark类型)
        forehead = landmarks[self.forehead_index]      # 额头: 10
        left_cheek = landmarks[self.left_cheek_index]  # 左脸颊: 234  
        chin = landmarks[self.chin_index]              # 下巴: 152
        right_cheek = landmarks[self.right_cheek_index] # 右脸颊: 454
        
        # 转换为numpy数组 (归一化坐标 [0,1])
        forehead_point = np.array([forehead.x, forehead.y, forehead.z])
        left_cheek_point = np.array([left_cheek.x, left_cheek.y, left_cheek.z])
        chin_point = np.array([chin.x, chin.y, chin.z])
        right_cheek_point = np.array([right_cheek.x, right_cheek.y, right_cheek.z])
        
        # 🔑 重要：应用X坐标修正（参考face_landmarker_cmaera_new.py）
        forehead_point[0] *= self.x_scale_factor
        left_cheek_point[0] *= self.x_scale_factor
        chin_point[0] *= self.x_scale_factor
        right_cheek_point[0] *= self.x_scale_factor
        
        # 调试信息（前几帧）
        if self.debug_mode and self.frame_count < 3:
            print(f"\n=== 帧 {self.frame_count} - 基于4点屏幕位置的变换 ===")
            print(f"总关键点数: {len(landmarks)}")
            print(f"宽高比修正系数: {self.x_scale_factor:.3f}")
            print(f"修正后关键点:")
            print(f"  额头[{self.forehead_index}]: {forehead_point}")
            print(f"  左脸颊[{self.left_cheek_index}]: {left_cheek_point}")
            print(f"  下巴[{self.chin_index}]: {chin_point}")
            print(f"  右脸颊[{self.right_cheek_index}]: {right_cheek_point}")
        
        # 🎯 基于4个landmarks点计算变换参数
        
        # 1. 计算面部尺寸和中心（归一化坐标）
        face_width = abs(right_cheek_point[0] - left_cheek_point[0])
        face_height = abs(chin_point[1] - forehead_point[1])
        face_center_x = (left_cheek_point[0] + right_cheek_point[0]) / 2
        face_center_y = (forehead_point[1] + chin_point[1]) / 2
        face_center_z = (forehead_point[2] + left_cheek_point[2] + chin_point[2] + right_cheek_point[2]) / 4
        
        # 2. 🔑 核心：基于landmarks点间距计算缩放
        # 从模型中获取对应4个点的距离作为参考
        model_face_width = np.linalg.norm(self.model_right_cheek - self.model_left_cheek)
        model_face_height = np.linalg.norm(self.model_chin - self.model_forehead)
        
        # 计算缩放因子：landmarks距离 / 模型距离
        # 🔧 重新设计缩放计算，让结果更合理
        # landmarks是归一化坐标，模型是mm单位，需要合适的比例转换
        
        # 设置合理的缩放基准：假设标准人脸在摄像头中的归一化尺寸
        reference_face_width = 0.35    # 🔧 增大参考尺寸让模型更小
        reference_face_height = 0.42   # 🔧 增大参考尺寸让模型更小
        
        # 基于实际检测尺寸与标准尺寸的比值计算缩放
        base_scale_x = face_width / reference_face_width
        base_scale_y = face_height / reference_face_height
        
        # 🔧 添加额外的缩小系数
        size_reduction = 0.8  # 整体缩小到80%
        
        scale_x = base_scale_x * size_reduction
        scale_y = base_scale_y * size_reduction
        scale_z = (scale_x + scale_y) / 2  # Z轴使用平均值
        
        # 🔧 限制缩放范围，避免过度变形
        scale_x = np.clip(scale_x, 0.1, 2.0)  
        scale_y = np.clip(scale_y, 0.1, 2.0)  
        scale_z = np.clip(scale_z, 0.1, 2.0)  
        
        # 3. 计算旋转角度
        # Roll角度：根据左右脸颊连线计算头部左右倾斜
        cheek_vector = right_cheek_point - left_cheek_point
        roll_angle = -np.arctan2(cheek_vector[1], cheek_vector[0])
        
        # Pitch角度：根据额头和下巴连线计算头部上下倾斜
        vertical_vector = chin_point - forehead_point
        pitch_angle = np.arctan2(vertical_vector[2], vertical_vector[1])
        
        # 🔧 修复Yaw角度：使用更准确的头部朝向计算
        # 方法1：基于左右脸颊的Z深度差，但增强幅度
        z_left = left_cheek_point[2]
        z_right = right_cheek_point[2]
        z_diff = z_right - z_left
        
        # 方法2：结合X坐标差异来增强Yaw检测
        # 当头向左转时，右脸颊会比左脸颊更靠近屏幕中心
        x_center = (left_cheek_point[0] + right_cheek_point[0]) / 2
        x_offset = face_center_x - x_center  # 面部中心相对于脸颊中心的偏移
        
        # 综合计算Yaw角度
        yaw_angle = np.arctan2(z_diff, face_width) * 2.0 + x_offset * 0.5  # 🔧 增强敏感度
        
        # 4. 坐标系转换 - 将归一化坐标转换为模型坐标
        screen_width = self.render_width
        screen_height = self.render_height
        
        # 转换为屏幕像素坐标
        screen_center_x = face_center_x * screen_width / self.x_scale_factor
        screen_center_y = face_center_y * screen_height
        
        # 转换为模型坐标系（以屏幕中心为原点）
        model_x = (screen_center_x - screen_width/2) * 0.05   
        model_y = -(screen_center_y - screen_height/2) * 0.05 + 1.5  # Y轴翻转 + 🔧 向上偏移1.5个单位
        model_z = face_center_z * 30 + 2  # Z轴适当前移
        
        # 调试信息
        if self.debug_mode and self.frame_count < 3:
            print(f"检测到的面部尺寸: 宽度={face_width:.4f}, 高度={face_height:.4f}")
            print(f"参考面部尺寸: 宽度={reference_face_width:.2f}, 高度={reference_face_height:.2f}")
            print(f"基础缩放: X={base_scale_x:.3f}, Y={base_scale_y:.3f}")
            print(f"缩小系数: {size_reduction}")
            print(f"最终缩放因子: X={scale_x:.3f}, Y={scale_y:.3f}, Z={scale_z:.3f}")
            print(f"Yaw计算: Z差值={z_diff:.4f}, X偏移={x_offset:.4f}")
            print(f"旋转角度: Roll={np.degrees(roll_angle):.1f}°, Pitch={np.degrees(pitch_angle):.1f}°, Yaw={np.degrees(yaw_angle):.1f}°")
            print(f"面部中心: ({face_center_x:.4f}, {face_center_y:.4f}, {face_center_z:.4f})")
            print(f"屏幕坐标: ({screen_center_x:.1f}, {screen_center_y:.1f})")
            print(f"模型坐标: ({model_x:.2f}, {model_y:.2f}, {model_z:.2f})")
        
        # 5. 构建变换矩阵：平移 + 旋转 + 缩放 (TRS变换)
        
        # 缩放矩阵
        scale_matrix = np.eye(4)
        scale_matrix[0, 0] = scale_x
        scale_matrix[1, 1] = scale_y
        scale_matrix[2, 2] = scale_z
        
        # 旋转矩阵 - 组合Roll、Pitch、Yaw旋转
        # Roll旋转（绕Z轴）
        cos_roll, sin_roll = np.cos(roll_angle), np.sin(roll_angle)
        roll_matrix = np.array([
            [cos_roll, -sin_roll, 0, 0],
            [sin_roll, cos_roll, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Pitch旋转（绕X轴）
        cos_pitch, sin_pitch = np.cos(pitch_angle), np.sin(pitch_angle)
        pitch_matrix = np.array([
            [1, 0, 0, 0],
            [0, cos_pitch, -sin_pitch, 0],
            [0, sin_pitch, cos_pitch, 0],
            [0, 0, 0, 1]
        ])
        
        # Yaw旋转（绕Y轴）
        cos_yaw, sin_yaw = np.cos(yaw_angle), np.sin(yaw_angle)
        yaw_matrix = np.array([
            [cos_yaw, 0, sin_yaw, 0],
            [0, 1, 0, 0],
            [-sin_yaw, 0, cos_yaw, 0],
            [0, 0, 0, 1]
        ])
        
        # 组合旋转矩阵：Yaw * Pitch * Roll
        rotation_matrix = yaw_matrix @ pitch_matrix @ roll_matrix
        
        # 平移矩阵
        translation_matrix = np.eye(4)
        translation_matrix[0, 3] = model_x
        translation_matrix[1, 3] = model_y
        translation_matrix[2, 3] = model_z
        
        # 最终变换矩阵：T * R * S（从右到左应用）
        transform_matrix = translation_matrix @ rotation_matrix @ scale_matrix
        
        # 🆕 保存变换矩阵供导出使用
        self.current_transform_matrix = transform_matrix
        
        if self.debug_mode and self.frame_count < 3:
            print(f"变换矩阵:\n{transform_matrix}")
        
        # 应用变换到模型顶点
        vertices = self.original_vertices.copy()
        vertices_homogeneous = np.hstack([vertices, np.ones((len(vertices), 1))])
        transformed_vertices = (transform_matrix @ vertices_homogeneous.T).T[:, :3]
        
        # 🆕 保存变换后的顶点供导出使用
        self.current_transformed_vertices = transformed_vertices
        
        # 🆕 新增：应用表情驱动
        if self.enable_expression_drive:
            transformed_vertices = self.apply_expression_drive(transformed_vertices, detection_result)
        
        # 更新模型
        self.face_mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
        self.face_mesh.compute_vertex_normals()
        
        self.frame_count += 1
        if self.frame_count >= 3:
            self.debug_mode = False
        
        return True
    
    def apply_expression_drive(self, vertices, detection_result):
        """🆕 基于BlendShapes应用表情驱动"""
        if not detection_result.face_blendshapes or len(detection_result.face_blendshapes) == 0:
            return vertices
        
        # 获取BlendShapes数据
        blendshapes = detection_result.face_blendshapes[0]
        
        # 创建顶点偏移数组
        vertex_offsets = np.zeros_like(vertices)
        
        # 遍历BlendShapes并应用到对应的顶点组
        for category in blendshapes:
            category_name = category.category_name
            score = category.score * self.expression_strength
            
            # 只处理有显著影响的BlendShapes
            if score < 0.1:
                continue
            
            # 🎭 嘴巴相关的BlendShapes
            if 'mouth' in category_name.lower() or 'jaw' in category_name.lower():
                self.apply_mouth_blendshapes(vertex_offsets, category_name, score)
            
            # 👁️ 眼睛相关的BlendShapes
            elif 'eye' in category_name.lower() or 'blink' in category_name.lower():
                self.apply_eye_blendshapes(vertex_offsets, category_name, score)
            
            # 🤔 眉毛相关的BlendShapes
            elif 'brow' in category_name.lower():
                self.apply_eyebrow_blendshapes(vertex_offsets, category_name, score)
            
            # 显示调试信息
            if self.show_blendshapes_debug and score > 0.2:
                print(f"   {category_name}: {score:.3f}")
        
        # 应用顶点偏移
        return vertices + vertex_offsets
    
    def apply_mouth_blendshapes(self, vertex_offsets, category_name, score):
        """应用嘴巴相关的BlendShapes"""
        mouth_indices = self.mouth_vertex_indices
        
        if 'open' in category_name.lower() or 'jaw' in category_name.lower():
            # 嘴巴张开：下巴向下移动
            for idx in mouth_indices:
                if idx < len(vertex_offsets):
                    vertex_offsets[idx, 1] -= score * 3.0  # Y轴向下
        
        elif 'smile' in category_name.lower():
            # 微笑：嘴角向上向外
            for idx in mouth_indices:
                if idx < len(vertex_offsets):
                    vertex_offsets[idx, 1] += score * 1.5  # Y轴向上
                    vertex_offsets[idx, 0] += score * 1.0 * np.sign(vertex_offsets[idx, 0])  # X轴向外
        
        elif 'pucker' in category_name.lower():
            # 撅嘴：嘴部向前
            for idx in mouth_indices:
                if idx < len(vertex_offsets):
                    vertex_offsets[idx, 2] += score * 2.0  # Z轴向前
    
    def apply_eye_blendshapes(self, vertex_offsets, category_name, score):
        """应用眼睛相关的BlendShapes"""
        if 'left' in category_name.lower():
            eye_indices = self.left_eye_vertex_indices
        elif 'right' in category_name.lower():
            eye_indices = self.right_eye_vertex_indices
        else:
            # 双眼
            eye_indices = self.left_eye_vertex_indices + self.right_eye_vertex_indices
        
        if 'blink' in category_name.lower() or 'close' in category_name.lower():
            # 眨眼：眼皮向内收缩
            for idx in eye_indices:
                if idx < len(vertex_offsets):
                    vertex_offsets[idx, 1] -= score * 1.0  # Y轴向内
        
        elif 'wide' in category_name.lower():
            # 眼睛睁大：向外扩张
            for idx in eye_indices:
                if idx < len(vertex_offsets):
                    vertex_offsets[idx, 1] += score * 0.8  # Y轴向外
    
    def apply_eyebrow_blendshapes(self, vertex_offsets, category_name, score):
        """应用眉毛相关的BlendShapes"""
        if 'left' in category_name.lower():
            brow_indices = self.eyebrow_vertex_indices[:len(self.eyebrow_vertex_indices)//2]
        elif 'right' in category_name.lower():
            brow_indices = self.eyebrow_vertex_indices[len(self.eyebrow_vertex_indices)//2:]
        else:
            brow_indices = self.eyebrow_vertex_indices
        
        if 'up' in category_name.lower() or 'raise' in category_name.lower():
            # 眉毛上扬
            for idx in brow_indices:
                if idx < len(vertex_offsets):
                    vertex_offsets[idx, 1] += score * 2.0  # Y轴向上
        
        elif 'down' in category_name.lower() or 'frown' in category_name.lower():
            # 眉毛下皱
            for idx in brow_indices:
                if idx < len(vertex_offsets):
                    vertex_offsets[idx, 1] -= score * 1.5  # Y轴向下
    
    def draw_original_landmarks(self, image, detection_result):
        """🆕 新增：绘制原始landmarks点和线框"""
        if not self.show_original_landmarks or not detection_result.face_landmarks:
            return image
        
        try:
            # 获取MediaPipe面部网格连接信息
            mp_face_mesh = mp.solutions.face_mesh
            
            for face_landmarks in detection_result.face_landmarks:
                height, width, _ = image.shape
                
                # 获取原始landmarks坐标（不做任何变换）
                coords = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks[:468]], dtype=np.float32)
                
                # 绘制landmarks点（绿色小圆点）
                for x_norm, y_norm, _ in coords:
                    x = int(x_norm * width)
                    y = int(y_norm * height)
                    # 确保坐标在有效范围内
                    if 0 <= x < width and 0 <= y < height:
                        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)  # 绿色点
                
                # 绘制面部网格连线（绿色细线）
                if hasattr(mp_face_mesh, 'FACEMESH_TESSELATION'):
                    connections = mp_face_mesh.FACEMESH_TESSELATION
                    for (start_idx, end_idx) in connections:
                        if start_idx < len(coords) and end_idx < len(coords):
                            sx = int(coords[start_idx, 0] * width)
                            sy = int(coords[start_idx, 1] * height)
                            ex = int(coords[end_idx, 0] * width)
                            ey = int(coords[end_idx, 1] * height)
                            # 确保坐标在有效范围内
                            if (0 <= sx < width and 0 <= sy < height and 
                                0 <= ex < width and 0 <= ey < height):
                                cv2.line(image, (sx, sy), (ex, ey), (0, 255, 0), 1)  # 绿色线
                
                # 🔑 特别标记4个关键点（红色大圆点）
                key_indices = [self.forehead_index, self.left_cheek_index, 
                              self.chin_index, self.right_cheek_index]
                key_labels = ["额头", "左脸颊", "下巴", "右脸颊"]
                
                for idx, label in zip(key_indices, key_labels):
                    if idx < len(coords):
                        x = int(coords[idx, 0] * width)
                        y = int(coords[idx, 1] * height)
                        if 0 <= x < width and 0 <= y < height:
                            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)  # 红色大点
                            cv2.putText(image, f"{idx}", (x+5, y-5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            return image
            
        except Exception as e:
            print(f"绘制原始landmarks错误: {e}")
            return image
    
    def export_realtime_model(self):
        """🆕 新增：导出当前实时变换后的3D模型"""
        if not hasattr(self, 'current_transformed_vertices') or self.current_transformed_vertices is None:
            print("❌ 没有可导出的变换后模型数据")
            return None
        
        try:
            # 生成唯一的文件名
            timestamp = int(time.time())
            filename = f"realtime_face_model_{timestamp}.obj"
            
            # 确保导出文件夹存在
            export_dir = "exported_models"
            os.makedirs(export_dir, exist_ok=True)
            filepath = os.path.join(export_dir, filename)
            
            print(f"\n=== 导出实时3D模型 ===")
            print(f"文件路径: {filepath}")
            print(f"顶点数: {len(self.current_transformed_vertices)}")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                # 写入OBJ文件头
                f.write("# FaceMatrixLab 实时3D面具模型\n")
                f.write(f"# 导出时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# 基于Andy_Wah_facemesh.obj，应用实时人脸跟踪变换\n")
                f.write(f"# 总顶点数: {len(self.current_transformed_vertices)}\n")
                f.write(f"# 总面数: {len(self.face_mesh.triangles)}\n")
                f.write("\n")
                
                # 写入变换后的顶点
                for i, vertex in enumerate(self.current_transformed_vertices):
                    f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
                
                f.write("\n")
                
                # 写入面信息（三角形）
                triangles = np.asarray(self.face_mesh.triangles)
                f.write("# 面定义 (三角形)\n")
                for i, triangle in enumerate(triangles):
                    # OBJ文件的顶点索引从1开始
                    f.write(f"f {triangle[0]+1} {triangle[1]+1} {triangle[2]+1}\n")
                
                # 如果有变换矩阵，也保存为注释
                if hasattr(self, 'current_transform_matrix'):
                    f.write("\n# 当前变换矩阵 (TRS)\n")
                    for row in self.current_transform_matrix:
                        f.write(f"# {row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f}\n")
            
            # 统计模型信息
            vertices = self.current_transformed_vertices
            print(f"变换后模型坐标范围:")
            print(f"  X: [{vertices[:, 0].min():.3f}, {vertices[:, 0].max():.3f}]")
            print(f"  Y: [{vertices[:, 1].min():.3f}, {vertices[:, 1].max():.3f}]")
            print(f"  Z: [{vertices[:, 2].min():.3f}, {vertices[:, 2].max():.3f}]")
            print(f"✅ 实时3D模型已导出: {filepath}")
            print("💡 可以在Blender中导入此OBJ文件查看效果")
            
            return filepath
            
        except Exception as e:
            print(f"❌ 导出实时模型失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run(self):
        """启动渲染器"""
        print("\n启动FaceMatrixLab 3D面具渲染器")
        print("=" * 60)
        print("🔑 使用MediaPipe faceLandmarks (468个NormalizedLandmark点)")
        print("🎯 重点使用4个特定点进行AR跟踪:")
        print(f"   额头: 索引 {self.forehead_index}")
        print(f"   左脸颊: 索引 {self.left_cheek_index}")
        print(f"   下巴: 索引 {self.chin_index}")
        print(f"   右脸颊: 索引 {self.right_cheek_index}")
        print("🚀 新增功能:")
        print("   ✅ 动态缩放：模型尺寸跟随人脸大小")
        print("   ✅ 头部旋转：Roll、Pitch、Yaw三轴旋转")
        print("   ✅ 完整变换：平移+旋转+缩放 (TRS)")
        print("   ✅ 16:9宽高比修正：正确处理1280x720分辨率")
        print("   ✅ 原始landmarks显示：绿色线框和关键点")
        print("   ✅ 表情驱动：基于BlendShapes的表情系统")
        print("=" * 60)
        print("控制说明:")
        print("  B键: 切换摄像机背景显示")
        print("  C键: 切换面具颜色")
        print("  1-6键: 直接选择面具颜色")
        print("  T键: 切换纹理贴图/统一颜色模式")
        print("  L键: 切换原始landmarks显示")
        print("  F键: 切换表情驱动功能")
        print("  D键: 切换BlendShapes调试信息")
        print("  +/-键: 调节表情强度 (0.0-3.0)")
        print("  E键: 导出当前实时3D模型为OBJ文件")
        print("  Q键: 退出程序")
        print("=" * 60)
        
        self.is_running = True
        
        # 设置可视化器
        if not self.setup_visualizer():
            print("❌ 可视化器初始化失败")
            return
        
        # 启动MediaPipe检测线程
        detection_thread = threading.Thread(target=self.detection_thread, daemon=True)
        detection_thread.start()
        
        # 等待检测线程启动
        time.sleep(2)
        
        # 创建AR合成窗口 - 1280x720
        cv2.namedWindow("AR Face Mask", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AR Face Mask", self.render_width, self.render_height)
        
        try:
            # 主渲染循环
            while self.is_running:
                # 读取最新数据
                frame = None
                try:
                    pkt = self.data_queue.get_nowait()
                    detection_result = pkt['detection_result']
                    frame = pkt.get('frame')
                    
                    # 保存最新帧用于AR合成
                    if frame is not None:
                        self.latest_camera_frame = frame.copy()
                    
                    # 更新3D模型
                    if self.update_face_model(detection_result):
                        self.vis.update_geometry(self.face_mesh)
                except queue.Empty:
                    pass
                
                # 离屏渲染获取3D模型图像
                self.vis.poll_events()
                self.vis.update_renderer()
                img_3d = np.asarray(self.vis.capture_screen_float_buffer(False))
                img_3d = (img_3d * 255).astype(np.uint8)
                img_3d_bgr = cv2.cvtColor(img_3d, cv2.COLOR_RGB2BGR)
                
                # AR合成: 将3D模型叠加到摄像机背景上
                if self.show_camera_background and self.latest_camera_frame is not None:
                    # 准备背景图像
                    bg = cv2.cvtColor(self.latest_camera_frame, cv2.COLOR_RGB2BGR)
                    bg = cv2.resize(bg, (self.render_width, self.render_height))
                    
                    # 🆕 新增：在背景上绘制原始landmarks
                    if hasattr(self, 'latest_detection_result'):
                        bg = self.draw_original_landmarks(bg, self.latest_detection_result)
                    
                    # 创建掩码: 非黑色像素的区域为3D模型
                    mask = img_3d_bgr.sum(axis=2) > 30
                    
                    # 合成图像: 背景 + 3D模型
                    composite = bg.copy()
                    composite[mask] = img_3d_bgr[mask]
                    
                    # 添加信息显示
                    fps_text = f"FPS: {self.current_fps:.1f}"
                    mask_text = f"面具颜色: {self.current_color_index+1}/{len(self.mask_colors)}"
                    landmarks_text = "faceLandmarks: 468点跟踪"
                    ratio_text = f"宽高比: {self.aspect_ratio:.3f} (16:9修正)"
                    cv2.putText(composite, fps_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(composite, mask_text, (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(composite, landmarks_text, (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(composite, ratio_text, (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    
                    # 显示原始landmarks状态
                    landmarks_status = f"原始landmarks: {'显示' if self.show_original_landmarks else '隐藏'} (L键切换)"
                    cv2.putText(composite, landmarks_status, (10, 190), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    # 🆕 显示表情驱动状态
                    expression_status = f"表情驱动: {'启用' if self.enable_expression_drive else '禁用'} (F键切换)"
                    cv2.putText(composite, expression_status, (10, 230), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    
                    # 显示表情强度
                    strength_status = f"表情强度: {self.expression_strength:.1f} (+-键调节)"
                    cv2.putText(composite, strength_status, (10, 270), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    
                    # 🆕 显示纹理状态
                    if hasattr(self, 'has_texture') and self.has_texture:
                        texture_status = f"渲染模式: {'纹理贴图' if self.texture_mode else '统一颜色'} (T键切换)"
                        cv2.putText(composite, texture_status, (10, 310), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    # 显示AR合成结果
                    cv2.imshow("AR Face Mask", composite)
                else:
                    # 只显示3D模型
                    cv2.imshow("AR Face Mask", img_3d_bgr)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('b'):
                    self.show_camera_background = not self.show_camera_background
                    print(f"背景显示: {'开启' if self.show_camera_background else '关闭'}")
                elif key == ord('c'):
                    # 循环切换下一个面具颜色
                    next_color = (self.current_color_index + 1) % len(self.mask_colors)
                    self.change_mask_color(next_color)
                elif key >= ord('1') and key <= ord('6'):
                    # 直接选择面具颜色 (1-6)
                    color_idx = key - ord('1')
                    if color_idx < len(self.mask_colors):
                        self.change_mask_color(color_idx)
                elif key == ord('l'):
                    # 🆕 新增：切换原始landmarks显示
                    self.show_original_landmarks = not self.show_original_landmarks
                    print(f"原始landmarks显示: {'开启' if self.show_original_landmarks else '关闭'}")
                elif key == ord('f'):
                    # 🆕 新增：切换表情驱动
                    self.enable_expression_drive = not self.enable_expression_drive
                    print(f"表情驱动: {'启用' if self.enable_expression_drive else '禁用'}")
                elif key == ord('d'):
                    # 🆕 新增：切换BlendShapes调试信息
                    self.show_blendshapes_debug = not self.show_blendshapes_debug
                    print(f"BlendShapes调试: {'显示' if self.show_blendshapes_debug else '隐藏'}")
                elif key == ord('+') or key == ord('='):
                    # 增加表情强度
                    self.expression_strength = min(self.expression_strength + 0.1, 3.0)
                    print(f"表情强度: {self.expression_strength:.1f}")
                elif key == ord('-'):
                    # 减少表情强度
                    self.expression_strength = max(self.expression_strength - 0.1, 0.0)
                    print(f"表情强度: {self.expression_strength:.1f}")
                elif key == ord('t'):
                    # 🆕 新增：切换纹理/颜色模式
                    self.toggle_texture_mode()
                elif key == ord('e'):
                    # 🆕 新增：导出当前实时3D模型
                    exported_file = self.export_realtime_model()
                    if exported_file:
                        print(f"🎉 实时3D模型已导出，可在Blender中查看: {exported_file}")
                    else:
                        print("❌ 导出失败，请确保有检测到的人脸")
                elif key == ord('q'):
                    break
                
                # 保存最新的检测结果用于landmarks绘制
                if 'detection_result' in locals():
                    self.latest_detection_result = detection_result
                
                # 控制帧率
                time.sleep(1.0 / self.fps_target)
                
        except KeyboardInterrupt:
            print("\n⏹️ 用户中断")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("🧹 清理资源...")
        self.is_running = False
        self.vis.destroy_window()
        cv2.destroyAllWindows()
        print("✅ 资源清理完成")


def main():
    """主函数"""
    print("FaceMatrixLab 3D 面具渲染器")
    print("使用MediaPipe faceLandmarks (468个NormalizedLandmark点) 实现精确跟踪")
    print("=" * 60)
    
    # 检查模型文件
    model_path = "obj/Andy_Wah_facemesh.obj"
    if not os.path.exists(model_path):
        print(f"❌ 错误：模型文件不存在 {model_path}")
        print("请确保Andy_Wah_facemesh.obj文件位于 obj/ 目录中")
        return
    
    try:
        # 创建并运行渲染器
        renderer = FaceMaskRenderer(camera_id=0, model_path=model_path)
        renderer.run()
        
    except Exception as e:
        print(f"❌ 程序运行错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("程序结束")


if __name__ == "__main__":
    main() 