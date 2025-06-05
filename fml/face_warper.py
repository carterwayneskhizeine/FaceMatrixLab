#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人脸像素级变形模块
使用Delaunay三角剖分和仿射变换实现实时人脸像素重映射 python = 3.9
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, List, Optional


class FaceWarper:
    """人脸像素级变形器"""
    
    def __init__(self):
        """初始化人脸变形器"""
        # 三角剖分缓存（首次调用时生成）
        self.cached_triangles = None
        self.mp_face_mesh = mp.solutions.face_mesh
        self._init_triangulation()
        print(f"FaceWarper初始化完成，有效三角形数量: {len(self.valid_triangles) if hasattr(self, 'valid_triangles') else 0}")
    
    def _init_triangulation(self):
        """初始化三角剖分数据"""
        try:
            # 尝试多种方式获取三角剖分数据
            triangulation_data = None
            
            # 方法1：尝试从 face_mesh 模块获取
            if hasattr(self.mp_face_mesh, 'FACEMESH_TRIANGULATION'):
                triangulation_data = self.mp_face_mesh.FACEMESH_TRIANGULATION
                print("使用 FACEMESH_TRIANGULATION")
            
            # 方法2：尝试从 face_mesh_connections 模块获取
            elif hasattr(self.mp_face_mesh, 'FACE_MESH_TRIANGULATION'):
                triangulation_data = self.mp_face_mesh.FACE_MESH_TRIANGULATION
                print("使用 FACE_MESH_TRIANGULATION")
                
            # 方法3：尝试直接从MediaPipe导入
            else:
                try:
                    from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TRIANGULATION
                    triangulation_data = FACEMESH_TRIANGULATION
                    print("从 face_mesh_connections 模块导入三角剖分数据")
                except ImportError:
                    try:
                        # 备用方案：使用经典的三角剖分数据
                        triangulation_data = self._get_default_triangulation()
                        print("使用默认三角剖分数据")
                    except Exception:
                        print("警告：无法获取任何三角剖分数据，尝试使用Delaunay自动生成")
                        triangulation_data = None
            
            if triangulation_data is not None:
                # 过滤出适用于468个landmarks的有效三角形
                all_triangles = list(triangulation_data)
                self.valid_triangles = []
                
                for triangle in all_triangles:
                    # 只保留所有顶点索引都小于468的三角形
                    if all(idx < 468 for idx in triangle):
                        self.valid_triangles.append(triangle)
                
                print(f"从{len(all_triangles)}个三角形中过滤出{len(self.valid_triangles)}个有效三角形")
            else:
                # 如果无法获取预定义的三角剖分，使用自动生成
                print("将在运行时自动生成Delaunay三角剖分")
                self.valid_triangles = []
                
        except Exception as e:
            print(f"初始化三角剖分失败: {e}")
            self.valid_triangles = []
            
    def _get_default_triangulation(self):
        """获取默认的三角剖分数据（备用方案）"""
        # 这里可以包含一些核心的三角形定义作为最后的备用方案
        # 为了简化，现在返回空列表，实际使用时可以添加手动定义的三角形
        return []
    
    def _generate_delaunay_triangulation(self, landmarks_pixels: np.ndarray) -> List[Tuple[int, int, int]]:
        """动态生成Delaunay三角剖分"""
        try:
            if len(landmarks_pixels) < 3:
                return []
                
            # 获取landmarks的外接矩形
            rect = cv2.boundingRect(landmarks_pixels.astype(np.float32))
            x, y, w, h = rect
            
            # 创建Subdiv2D对象
            subdiv = cv2.Subdiv2D((x, y, x+w, y+h))
            
            # 添加所有landmarks点
            point_to_index = {}
            for i, (px, py) in enumerate(landmarks_pixels[:, :2]):
                try:
                    subdiv.insert((float(px), float(py)))
                    point_to_index[(float(px), float(py))] = i
                except cv2.error:
                    continue
            
            # 获取三角形
            triangles = []
            triangle_list = subdiv.getTriangleList()
            
            for t in triangle_list:
                # 每个三角形包含6个坐标值：x1,y1,x2,y2,x3,y3
                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])
                
                # 查找对应的索引
                try:
                    idx1 = point_to_index.get(pt1, -1)
                    idx2 = point_to_index.get(pt2, -1)
                    idx3 = point_to_index.get(pt3, -1)
                    
                    if idx1 != -1 and idx2 != -1 and idx3 != -1:
                        triangles.append((idx1, idx2, idx3))
                except:
                    continue
                    
            print(f"动态生成了 {len(triangles)} 个Delaunay三角形")
            return triangles
            
        except Exception as e:
            print(f"生成Delaunay三角剖分失败: {e}")
            return []
    
    def normalize_to_pixel_coords(self, normalized_landmarks: np.ndarray, 
                                width: int, height: int) -> np.ndarray:
        """将归一化的landmarks转换为像素坐标"""
        pixel_coords = np.zeros((len(normalized_landmarks), 2), dtype=np.float32)
        for i, (x_norm, y_norm, _) in enumerate(normalized_landmarks):
            pixel_coords[i] = [x_norm * width, y_norm * height]
        return pixel_coords
    
    def get_face_roi(self, landmarks_pixels: np.ndarray, 
                    width: int, height: int, padding: int = 50) -> Tuple[int, int, int, int]:
        """获取人脸区域的边界框"""
        if len(landmarks_pixels) == 0:
            return 0, 0, width, height
        
        x_min = max(0, int(np.min(landmarks_pixels[:, 0])) - padding)
        y_min = max(0, int(np.min(landmarks_pixels[:, 1])) - padding)
        x_max = min(width, int(np.max(landmarks_pixels[:, 0])) + padding)
        y_max = min(height, int(np.max(landmarks_pixels[:, 1])) + padding)
        
        return x_min, y_min, x_max - x_min, y_max - y_min
    
    def create_triangle_mask(self, triangle_points: np.ndarray, 
                           roi_offset: Tuple[int, int], 
                           roi_size: Tuple[int, int]) -> np.ndarray:
        """创建三角形掩码"""
        mask = np.zeros(roi_size, dtype=np.uint8)
        
        # 将三角形坐标转换为相对于ROI的坐标
        triangle_roi = triangle_points.copy()
        triangle_roi[:, 0] -= roi_offset[0]
        triangle_roi[:, 1] -= roi_offset[1]
        
        # 确保坐标在ROI范围内
        triangle_roi = np.clip(triangle_roi, 0, [roi_size[0]-1, roi_size[1]-1])
        
        try:
            cv2.fillConvexPoly(mask, np.int32(triangle_roi), 255)
        except Exception as e:
            print(f"创建三角形掩码失败: {e}")
            
        return mask
    
    def warp_triangle(self, source_image: np.ndarray,
                     src_triangle: np.ndarray,
                     dst_triangle: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int,int,int,int]]:
        """对单个三角形进行ROI仿射变换"""
        try:
            # 源三角形和目标三角形的边界框
            src_rect = cv2.boundingRect(src_triangle.astype(np.float32))  # x,y,w,h
            dst_rect = cv2.boundingRect(dst_triangle.astype(np.float32))
            x0, y0, w0, h0 = src_rect
            x1, y1, w1, h1 = dst_rect
            # 确保有效
            if w0 <= 0 or h0 <= 0 or w1 <= 0 or h1 <= 0:
                return None, None, None
            # 局部三角形坐标
            src_tri_local = src_triangle - np.array([x0, y0], dtype=np.float32)
            dst_tri_local = dst_triangle - np.array([x1, y1], dtype=np.float32)
            # ROI裁剪源图
            src_roi = source_image[y0:y0+h0, x0:x0+w0]
            # 计算仿射矩阵
            M = cv2.getAffineTransform(
                src_tri_local.astype(np.float32),
                dst_tri_local.astype(np.float32)
            )
            # 对ROI区域进行仿射变换
            warped_patch = cv2.warpAffine(
                src_roi, M, (w1, h1),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )
            # 创建三角形掩码（ROI大小）
            mask = np.zeros((h1, w1), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(dst_tri_local), 255)
            return warped_patch, mask, (x1, y1, w1, h1)
        except Exception as e:
            print(f"三角形变形失败: {e}")
            return None, None, None
    
    def _is_triangle_flipped(self, src_triangle: np.ndarray, dst_triangle: np.ndarray) -> bool:
        """
        判断三角形在投影中是否翻转（法线方向前后相反）
        通过比较源三角形与目标三角形的有向面积符号是否相反
        """
        # 计算源三角形有向面积（叉积）
        cross_src = ((src_triangle[1][0] - src_triangle[0][0]) * (src_triangle[2][1] - src_triangle[0][1])
                     - (src_triangle[1][1] - src_triangle[0][1]) * (src_triangle[2][0] - src_triangle[0][0]))
        # 计算目标三角形有向面积（叉积）
        cross_dst = ((dst_triangle[1][0] - dst_triangle[0][0]) * (dst_triangle[2][1] - dst_triangle[0][1])
                     - (dst_triangle[1][1] - dst_triangle[0][1]) * (dst_triangle[2][0] - dst_triangle[0][0]))
        # 如果符号相反，则三角形翻转
        return cross_src * cross_dst < 0

    def warp_face_texture(self, source_image: np.ndarray,
                         src_landmarks_pixels: np.ndarray,
                         dst_landmarks_pixels: np.ndarray,
                         dst_landmarks_3d: np.ndarray = None,
                         enable_wireframe: bool = True,
                         use_lambert_material: bool = False,
                         face_only_mode: bool = False,
                         wireframe_thickness: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        核心方法：将源人脸纹理变形到目标形状
        
        Args:
            source_image: 源图像
            src_landmarks_pixels: 源landmarks像素坐标 (N, 2)
            dst_landmarks_pixels: 目标landmarks像素坐标 (N, 2)
            dst_landmarks_3d: 目标landmarks的3D坐标 (N, 3)，包含真实的Z值
            enable_wireframe: 是否启用黑色线框显示
            use_lambert_material: 是否使用Lambert材质渲染而非复制纹理
            face_only_mode: 是否只显示面部模型，背景纯黑
            wireframe_thickness: 线框粗细
        
        Returns:
            warped_texture: 变形后的纹理图像（带线框）
            face_mask: 人脸区域掩码
        """
        # 确定要使用的三角剖分
        if len(self.valid_triangles) == 0:
            # 如果没有预定义的三角剖分，使用缓存的Delaunay三角剖分
            if self.cached_triangles is None:
                self.cached_triangles = self._generate_delaunay_triangulation(src_landmarks_pixels)
                print(f"缓存生成了 {len(self.cached_triangles)} 个Delaunay三角形")
            use_triangles = self.cached_triangles
        else:
            use_triangles = self.valid_triangles
        
        height, width = source_image.shape[:2]
        
        # 初始化输出图像和总掩码
        if face_only_mode:
            # 面部专用模式：背景纯黑
            warped_texture = np.zeros_like(source_image)
        else:
            # 普通模式：背景复制原图
            warped_texture = np.zeros_like(source_image)
        
        total_face_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Lambert材质参数 - 使用和旧版本相同的简单参数
        if use_lambert_material:
            base_color = np.array([255, 255, 255], dtype=np.float32)  # 纯白色基础色
            light_direction = np.array([0.5, -0.8, 0.3], dtype=np.float32)  # 光源方向 (从左上前方照射)
            light_direction = light_direction / np.linalg.norm(light_direction)  # 归一化
            ambient_intensity = 0.7  # 环境光强度 0.7
            diffuse_intensity = 0.7  # 漫反射强度 0.7
        
        # 线框参数
        wireframe_color = (0, 0, 0)  # 黑色线框
        wireframe_alpha = 0.7        # 线条透明度（0.0-1.0，用于模拟细线效果）
        
        # 逐三角形处理
        triangle_count = 0
        wireframe_triangles = []  # 存储需要绘制线框的三角形
        
        for triangle_indices in use_triangles:
            # 获取源和目标三角形顶点
            src_triangle = src_landmarks_pixels[list(triangle_indices)]
            dst_triangle = dst_landmarks_pixels[list(triangle_indices)]
            # 跳过投影中翻转的三角形（避免前后面重叠）
            if self._is_triangle_flipped(src_triangle, dst_triangle):
                continue
            try:
                src_triangle = src_landmarks_pixels[list(triangle_indices)]
                dst_triangle = dst_landmarks_pixels[list(triangle_indices)]
                
                # 检查三角形是否退化（面积太小）
                if cv2.contourArea(src_triangle.astype(np.float32)) < 1.0 or \
                   cv2.contourArea(dst_triangle.astype(np.float32)) < 1.0:
                    continue
                
                # 如果使用Lambert材质，直接渲染灰色三角形
                if use_lambert_material:
                    # 获取目标三角形的3D坐标（假设z=0平面上）
                    triangle_3d = np.column_stack([dst_triangle, np.zeros(3)])
                    
                    # 计算Lambert光照颜色 - 使用简单的渲染方法
                    triangle_color = self._render_lambert_triangle_simple(
                        triangle_3d, base_color, light_direction, 
                        ambient_intensity, diffuse_intensity
                    )
                    
                    # 创建单色填充的三角形
                    dst_rect = cv2.boundingRect(dst_triangle.astype(np.float32))
                    x, y, w, h = dst_rect
                    
                    if w <= 0 or h <= 0:
                        continue
                    
                    # 创建填充的三角形patch
                    warped_patch = np.full((h, w, 3), triangle_color, dtype=np.uint8)
                    
                    # 创建三角形掩码
                    mask_patch = np.zeros((h, w), dtype=np.uint8)
                    dst_tri_local = dst_triangle - np.array([x, y], dtype=np.float32)
                    cv2.fillConvexPoly(mask_patch, np.int32(dst_tri_local), 255)
                    
                    rect = (x, y, w, h)
                else:
                    # 【恢复】原始纹理复制方式 - 昂贵的操作
                    warped_patch, mask_patch, rect = self.warp_triangle(
                        source_image, src_triangle, dst_triangle
                    )
                    if warped_patch is None:
                        continue
                
                x, y, w, h = rect
                # 合并到输出图像
                mask_3c = cv2.merge([mask_patch, mask_patch, mask_patch])
                mask_norm = mask_3c.astype(np.float32) / 255.0
                # 只处理ROI区域
                roi_region = warped_texture[y:y+h, x:x+w]
                
                if face_only_mode:
                    # 面部专用模式：直接替换到黑色背景上
                    roi_region = (roi_region * (1 - mask_norm) + warped_patch * mask_norm).astype(np.uint8)
                else:
                    # 普通模式：正常混合
                    roi_region = (roi_region * (1 - mask_norm) + warped_patch * mask_norm).astype(np.uint8)
                
                warped_texture[y:y+h, x:x+w] = roi_region
                # 更新总掩码ROI
                existing_mask = total_face_mask[y:y+h, x:x+w]
                total_face_mask[y:y+h, x:x+w] = cv2.bitwise_or(existing_mask, mask_patch)
                
                # 保存三角形信息用于线框绘制（只在Lambert材质模式下）
                if enable_wireframe and use_lambert_material:
                    wireframe_triangles.append(dst_triangle)
                
                triangle_count += 1
            except Exception as e:
                print(f"处理三角形 {triangle_indices} 时出错: {e}")
                continue
        
        # 绘制线框（只在Lambert材质模式下且启用线框时）
        if enable_wireframe and use_lambert_material and len(wireframe_triangles) > 0:
            self._draw_wireframe(warped_texture, wireframe_triangles, wireframe_color, wireframe_thickness, wireframe_alpha)
        
        render_mode = "Lambert材质" if use_lambert_material else "原始纹理复制"
        if face_only_mode:
            render_mode += " (面部专用)"
        print(f"成功处理 {triangle_count}/{len(use_triangles)} 个三角形 (模式: {render_mode})")
        if enable_wireframe and use_lambert_material:
            print(f"绘制了 {len(wireframe_triangles)} 个三角形的线框")
        return warped_texture, total_face_mask
    
    def apply_gaussian_blur_edge(self, original_frame: np.ndarray,
                                warped_texture: np.ndarray,
                                original_mask: np.ndarray,
                                warped_mask: np.ndarray,
                                filter_diameter: int = 15) -> np.ndarray:
        """
        在变形后扩展出来的新区域应用双边滤波，保持脸部内部清晰
        
        Args:
            original_frame: 原始图像
            warped_texture: 变形后的纹理
            original_mask: 原始人脸掩码
            warped_mask: 变形后人脸掩码
            filter_diameter: 双边滤波直径
            
        Returns:
            filtered_result: 应用双边滤波后的结果
        """
        try:
            # 找到扩展区域：变形后区域 - 原始区域（新增的区域）
            expanded_mask = cv2.bitwise_and(warped_mask, cv2.bitwise_not(original_mask))
            
            # 创建需要滤波的区域掩码（扩展区域）
            blur_region_mask = expanded_mask
            
            if np.sum(blur_region_mask) == 0:
                return warped_texture
            
            # 对扩展区域应用双边滤波，保持原脸部区域清晰
            result_texture = warped_texture.copy()
            
            # 使用双边滤波：保持边缘的同时平滑纹理
            # 参数：直径, sigmaColor(颜色相似性), sigmaSpace(空间距离)
            filtered_texture = cv2.bilateralFilter(warped_texture, filter_diameter, 80, 80)
            
            # 创建软边缘掩码，使用形态学操作避免硬边界
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            blur_mask_smooth = cv2.morphologyEx(blur_region_mask, cv2.MORPH_CLOSE, kernel)
            blur_mask_smooth = cv2.morphologyEx(blur_mask_smooth, cv2.MORPH_OPEN, kernel)
            
            # 使用距离变换创建渐变掩码
            dist_transform = cv2.distanceTransform(blur_mask_smooth, cv2.DIST_L2, 5)
            dist_transform = cv2.normalize(dist_transform, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            
            # 应用平滑函数使过渡更自然
            smooth_mask = np.power(dist_transform, 0.5)  # 平方根函数使过渡更平缓
            
            # 只在扩展区域应用滤波，保持原脸部区域不变
            smooth_mask_3c = cv2.merge([smooth_mask, smooth_mask, smooth_mask])
            
            # 在扩展区域中：原图像 = 清晰纹理，滤波区域 = 双边滤波纹理
            result_texture = (warped_texture * (1 - smooth_mask_3c) + 
                            filtered_texture * smooth_mask_3c).astype(np.uint8)
            
            return result_texture
            
        except Exception as e:
            print(f"应用双边滤波边缘失败: {e}")
            return warped_texture
    
    def apply_face_warp(self, original_frame: np.ndarray,
                       src_landmarks_normalized: np.ndarray,
                       dst_landmarks_normalized: np.ndarray,
                       blend_ratio: float = 1.0,
                       enable_blur_edge: bool = False,
                       enable_wireframe: bool = True,
                       use_lambert_material: bool = False,
                       face_only_mode: bool = False,
                       wireframe_thickness: float = 0.2) -> np.ndarray:
        """
        应用人脸变形到原始帧（支持Lambert材质渲染+线框或原始纹理复制）
        
        Args:
            original_frame: 原始图像帧
            src_landmarks_normalized: 源landmarks（归一化坐标）
            dst_landmarks_normalized: 目标landmarks（归一化坐标）
            blend_ratio: 混合比例 (0-1, 1为完全替换)
            enable_blur_edge: 是否启用边缘高斯模糊
            enable_wireframe: 是否启用黑色线框显示
            use_lambert_material: 是否使用Lambert材质渲染而非复制纹理
            face_only_mode: 是否只显示面部模型，背景纯黑
            wireframe_thickness: 线框粗细
        
        Returns:
            result_frame: 应用变形后的图像
        """
        height, width = original_frame.shape[:2]
        
        # 转换为像素坐标
        src_pixels = self.normalize_to_pixel_coords(src_landmarks_normalized, width, height)
        dst_pixels = self.normalize_to_pixel_coords(dst_landmarks_normalized, width, height)
        
        # 获取原始人脸掩码
        _, original_face_mask = self.warp_face_texture(original_frame, src_pixels, src_pixels)
        
        # 执行人脸变形，传递3D坐标、线框和纹理模式参数
        warped_texture, face_mask = self.warp_face_texture(
            original_frame, src_pixels, dst_pixels, dst_landmarks_normalized, 
            enable_wireframe, use_lambert_material, face_only_mode, wireframe_thickness
        )
        
        # 如果启用边缘模糊，应用高斯模糊
        if enable_blur_edge:
            warped_texture = self.apply_gaussian_blur_edge(
                original_frame, warped_texture, original_face_mask, face_mask
            )
        
        # 创建结果图像
        if face_only_mode:
            # 面部专用模式：直接使用warped_texture（背景已经是黑色）
            result_frame = warped_texture.copy()
        else:
            # 普通模式：与原始图像混合
            result_frame = original_frame.copy()
            
            if np.sum(face_mask) > 0:  # 确保有有效的人脸区域
                # 应用混合
                mask_3channel = cv2.merge([face_mask, face_mask, face_mask]).astype(np.float32) / 255.0
                mask_3channel *= blend_ratio
                
                result_frame = (original_frame * (1 - mask_3channel) + 
                              warped_texture * mask_3channel).astype(np.uint8)
        
        return result_frame
    
    def smooth_landmarks(self, current_landmarks: np.ndarray, 
                        previous_landmarks: Optional[np.ndarray],
                        smoothing_factor: float = 0.7) -> np.ndarray:
        """
        平滑landmarks以减少抖动
        
        Args:
            current_landmarks: 当前帧的landmarks
            previous_landmarks: 前一帧的landmarks
            smoothing_factor: 平滑系数 (0-1, 越大越平滑)
        
        Returns:
            smoothed_landmarks: 平滑后的landmarks
        """
        if previous_landmarks is None:
            return current_landmarks
        
        return (smoothing_factor * previous_landmarks + 
                (1 - smoothing_factor) * current_landmarks) 

    def _calculate_triangle_normal(self, triangle_3d: np.ndarray) -> np.ndarray:
        """
        计算三角形的法向量（用于Lambert光照）
        
        Args:
            triangle_3d: 三角形的3D坐标 (3, 3) [v1, v2, v3]
        
        Returns:
            normal: 归一化的法向量 (3,)
        """
        try:
            # 计算两个边向量
            edge1 = triangle_3d[1] - triangle_3d[0]  # v2 - v1
            edge2 = triangle_3d[2] - triangle_3d[0]  # v3 - v1
            
            # 计算叉积得到法向量
            normal = np.cross(edge1, edge2)
            
            # 归一化
            norm_length = np.linalg.norm(normal)
            if norm_length > 1e-8:
                normal = normal / norm_length
            else:
                normal = np.array([0.0, 0.0, 1.0])  # 默认向前的法向量
            
            # 确保法向量指向观察者（Z分量为正）
            if normal[2] < 0:
                normal = -normal
            
            return normal
        except Exception as e:
            print(f"计算法向量失败: {e}")
            return np.array([0.0, 0.0, 1.0])  # 默认向前的法向量

    def _render_lambert_triangle(self, triangle_3d: np.ndarray, 
                                base_color: np.ndarray, 
                                main_light_direction: np.ndarray,
                                ambient_intensity: float = 0.7, 
                                main_diffuse_intensity: float = 0.8,
                                front_light_direction: np.ndarray = None,
                                front_diffuse_intensity: float = 0.4) -> np.ndarray:
        """
        使用ZBrush MatCap风格的双光源光照模型渲染三角形颜色
        
        Args:
            triangle_3d: 三角形3D坐标 (3, 3)
            base_color: 基础色 (3,) RGB
            main_light_direction: 主光源方向 (3,)
            ambient_intensity: 环境光强度
            main_diffuse_intensity: 主光源漫反射强度
            front_light_direction: 正面补光方向 (3,)
            front_diffuse_intensity: 正面补光强度
        
        Returns:
            final_color: 最终颜色 (3,) RGB
        """
        try:
            # 计算三角形法向量
            normal = self._calculate_triangle_normal(triangle_3d)
            
            # 主光源光照强度计算
            main_light_intensity = max(0.0, np.dot(normal, main_light_direction))
            main_light_intensity = np.power(main_light_intensity, 0.7)  # 非线性映射
            
            # 正面补光光照强度计算
            if front_light_direction is None:
                front_light_direction = np.array([0.0, 0.0, 1.0])  # 默认正面方向
            
            front_light_intensity = max(0.0, np.dot(normal, front_light_direction))
            front_light_intensity = np.power(front_light_intensity, 0.8)  # 稍微不同的非线性映射
            
            # 添加轻微的边缘光效果（Fresnel-like）
            view_direction = np.array([0.0, 0.0, 1.0])  # 假设视线方向为正Z
            fresnel_factor = 1.0 - abs(np.dot(normal, view_direction))
            rim_light = fresnel_factor * 0.2  # 轻微的边缘光
            
            # 组合所有光照分量
            total_intensity = (ambient_intensity + 
                             main_diffuse_intensity * main_light_intensity + 
                             front_diffuse_intensity * front_light_intensity + 
                             rim_light)
            
            # ZBrush MatCap风格的亮度映射 - 更高的对比度
            total_intensity = np.clip(total_intensity, 0.0, 1.4)  # 允许更高的亮度范围
            
            # 使用S曲线调整对比度，模拟MatCap的视觉效果
            if total_intensity > 0.5:
                # 亮部增强
                total_intensity = 0.5 + (total_intensity - 0.5) * 1.2
            else:
                # 暗部保持
                total_intensity = total_intensity * 0.9
            
            total_intensity = min(1.0, total_intensity)  # 最终限制在1.0
            
            # 计算最终颜色
            final_color = base_color * total_intensity
            final_color = np.clip(final_color, 0, 255).astype(np.uint8)
            
            return final_color
        except Exception as e:
            print(f"双光源MatCap渲染失败: {e}")
            return base_color.astype(np.uint8)  # 返回基础色 

    def _render_lambert_triangle_simple(self, triangle_3d: np.ndarray, 
                                        base_color: np.ndarray, 
                                        light_direction: np.ndarray,
                                        ambient_intensity: float = 0.7, 
                                        diffuse_intensity: float = 0.7) -> np.ndarray:
        """
        使用简单的Lambert光照模型渲染三角形颜色（和旧版本相同的逻辑）
        
        Args:
            triangle_3d: 三角形3D坐标 (3, 3)
            base_color: 基础色 (3,) RGB
            light_direction: 光源方向 (3,)
            ambient_intensity: 环境光强度
            diffuse_intensity: 漫反射强度
        
        Returns:
            final_color: 最终颜色 (3,) RGB
        """
        try:
            # 计算三角形法向量
            normal = self._calculate_triangle_normal(triangle_3d)
            
            # 计算光照强度 (Lambert漫反射)
            # dot(normal, -light_direction) 因为光源方向是指向光源的
            light_intensity = max(0.0, np.dot(normal, -light_direction))
            
            # 组合环境光和漫反射光
            total_intensity = ambient_intensity + diffuse_intensity * light_intensity
            total_intensity = min(1.0, total_intensity)  # 限制在1.0以内
            
            # 计算最终颜色
            final_color = base_color * total_intensity
            final_color = np.clip(final_color, 0, 255).astype(np.uint8)
            
            return final_color
        except Exception as e:
            print(f"简单Lambert渲染失败: {e}")
            return base_color.astype(np.uint8)  # 返回基础色

    def _draw_wireframe(self, image: np.ndarray, triangles: List[np.ndarray], 
                       color: Tuple[int, int, int] = (0, 0, 0), thickness: float = 1.0, alpha: float = 0.7):
        """
        在图像上绘制三角形线框（支持亚像素级细线）
        
        Args:
            image: 目标图像
            triangles: 三角形列表，每个三角形是(3, 2)的坐标数组
            color: 线框颜色 (B, G, R)
            thickness: 线条粗细（支持小数，如0.2）
            alpha: 线条透明度（0.0-1.0）
        """
        try:
            # 为了支持亚像素级细线，我们使用透明度叠加的方式
            if thickness < 1.0:
                # 对于细线，使用1像素宽度但降低透明度来模拟细线效果
                actual_thickness = 1
                effective_alpha = alpha * thickness  # 线条越细，透明度越低
            else:
                actual_thickness = int(thickness)
                effective_alpha = alpha
            
            # 创建线框层
            overlay = image.copy()
            
            for triangle in triangles:
                # 将浮点坐标转换为整数像素坐标
                pts = np.int32(triangle)
                
                # 检查坐标是否在图像范围内
                height, width = image.shape[:2]
                valid_pts = []
                for pt in pts:
                    x, y = pt
                    if 0 <= x < width and 0 <= y < height:
                        valid_pts.append(pt)
                
                # 只有当所有点都有效时才绘制
                if len(valid_pts) == 3:
                    # 绘制三角形的三条边，使用抗锯齿效果
                    cv2.line(overlay, tuple(pts[0]), tuple(pts[1]), color, actual_thickness, cv2.LINE_AA)
                    cv2.line(overlay, tuple(pts[1]), tuple(pts[2]), color, actual_thickness, cv2.LINE_AA)
                    cv2.line(overlay, tuple(pts[2]), tuple(pts[0]), color, actual_thickness, cv2.LINE_AA)
            
            # 将线框层与原图像混合，实现透明度效果
            cv2.addWeighted(overlay, effective_alpha, image, 1 - effective_alpha, 0, image)
                    
        except Exception as e:
            print(f"绘制线框失败: {e}") 