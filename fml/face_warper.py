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
                         dst_landmarks_pixels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        核心方法：将源人脸纹理变形到目标形状
        
        Args:
            source_image: 源图像
            src_landmarks_pixels: 源landmarks像素坐标 (N, 2)
            dst_landmarks_pixels: 目标landmarks像素坐标 (N, 2)
        
        Returns:
            warped_texture: 变形后的纹理图像
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
        warped_texture = np.zeros_like(source_image)
        total_face_mask = np.zeros((height, width), dtype=np.uint8)
        
        # 逐三角形处理
        triangle_count = 0
        
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
                
                # 使用原始纹理复制方式
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
                
                # 正常混合
                roi_region = (roi_region * (1 - mask_norm) + warped_patch * mask_norm).astype(np.uint8)
                
                warped_texture[y:y+h, x:x+w] = roi_region
                # 更新总掩码ROI
                existing_mask = total_face_mask[y:y+h, x:x+w]
                total_face_mask[y:y+h, x:x+w] = cv2.bitwise_or(existing_mask, mask_patch)
                
                triangle_count += 1
            except Exception as e:
                print(f"处理三角形 {triangle_indices} 时出错: {e}")
                continue
        
        print(f"成功处理 {triangle_count}/{len(use_triangles)} 个三角形")
        return warped_texture, total_face_mask
    
    def apply_face_warp(self, original_frame: np.ndarray,
                       src_landmarks_normalized: np.ndarray,
                       dst_landmarks_normalized: np.ndarray,
                       blend_ratio: float = 1.0) -> np.ndarray:
        """
        应用人脸变形到原始帧
        
        Args:
            original_frame: 原始图像帧
            src_landmarks_normalized: 源landmarks（归一化坐标）
            dst_landmarks_normalized: 目标landmarks（归一化坐标）
            blend_ratio: 混合比例 (0-1, 1为完全替换)
        
        Returns:
            result_frame: 应用变形后的图像
        """
        height, width = original_frame.shape[:2]
        
        # 转换为像素坐标
        src_pixels = self.normalize_to_pixel_coords(src_landmarks_normalized, width, height)
        dst_pixels = self.normalize_to_pixel_coords(dst_landmarks_normalized, width, height)
        
        # 获取原始人脸掩码
        _, original_face_mask = self.warp_face_texture(original_frame, src_pixels, src_pixels)
        
        # 执行人脸变形
        warped_texture, face_mask = self.warp_face_texture(
            original_frame, src_pixels, dst_pixels
        )
        
        # 与原始图像混合
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