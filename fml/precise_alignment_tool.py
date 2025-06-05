#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
精确面部模型对齐工具
基于canonical_face_model.obj的准确分析结果进行对齐
"""

import os
import sys
import numpy as np
from scipy.optimize import minimize

class PreciseFaceAlignmentTool:
    def __init__(self):
        self.vertices = []
        self.texture_coords = []
        self.faces = []
        self.comments = []
        
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
        
        # 目标几何中心（基于canonical模型）
        self.target_center = np.array([-0.000000, -0.676437, 4.158965])
        
        # 目标鼻尖位置（来自canonical模型）
        self.target_nose_tip = np.array([0.000000, -0.463170, 7.586580])
        
    def load_obj_file(self, obj_path):
        """加载OBJ文件"""
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"OBJ文件不存在: {obj_path}")
            
        print(f"正在加载OBJ文件: {obj_path}")
        
        with open(obj_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line.startswith('v '):
                    # 顶点坐标
                    parts = line.split()
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    self.vertices.append([x, y, z])
                elif line.startswith('vt '):
                    # 纹理坐标
                    self.texture_coords.append(line)
                elif line.startswith('f '):
                    # 面
                    self.faces.append(line)
                elif line.startswith('#'):
                    # 注释
                    self.comments.append(line)
                    
        print(f"加载完成: {len(self.vertices)} 个顶点, {len(self.faces)} 个面")
        self.vertices = np.array(self.vertices)
        
    def compute_precise_alignment_transform(self):
        """计算精确的对齐变换"""
        print("计算精确对齐变换...")
        
        def create_transform_matrix(params):
            """根据参数创建变换矩阵"""
            # 参数: [tx, ty, tz, rx, ry, rz, scale]
            tx, ty, tz, rx, ry, rz, scale = params
            
            # 旋转矩阵
            cos_x, sin_x = np.cos(rx), np.sin(rx)
            cos_y, sin_y = np.cos(ry), np.sin(ry)
            cos_z, sin_z = np.cos(rz), np.sin(rz)
            
            Rx = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
            Ry = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
            Rz = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
            
            R = Rz @ Ry @ Rx
            
            # 缩放
            R = R * scale
            
            # 平移
            translation = np.array([tx, ty, tz])
            
            return R, translation
        
        def apply_transform(vertices, rotation_matrix, translation):
            """应用变换"""
            return (rotation_matrix @ vertices.T).T + translation
        
        def objective_function(params):
            """优化目标函数"""
            R, t = create_transform_matrix(params)
            transformed_vertices = apply_transform(self.vertices, R, t)
            
            total_cost = 0
            
            # 目标1：中线点的X坐标应该为0
            centerline_cost = 0
            valid_centerline_count = 0
            for idx in self.centerline_indices:
                if idx < len(transformed_vertices):
                    centerline_cost += abs(transformed_vertices[idx, 0]) ** 2
                    valid_centerline_count += 1
            
            if valid_centerline_count > 0:
                centerline_cost = centerline_cost / valid_centerline_count
                total_cost += 100 * centerline_cost  # 高权重
            
            # 目标2：对称点对应该对称
            symmetry_cost = 0
            valid_pairs_count = 0
            for left_idx, right_idx in self.symmetric_pairs:
                if left_idx < len(transformed_vertices) and right_idx < len(transformed_vertices):
                    left_point = transformed_vertices[left_idx]
                    right_point = transformed_vertices[right_idx]
                    
                    # X坐标应该相反
                    x_error = (left_point[0] + right_point[0]) ** 2
                    # Y和Z坐标应该相同
                    y_error = (left_point[1] - right_point[1]) ** 2
                    z_error = (left_point[2] - right_point[2]) ** 2
                    
                    symmetry_cost += x_error + y_error + z_error
                    valid_pairs_count += 1
            
            if valid_pairs_count > 0:
                symmetry_cost = symmetry_cost / valid_pairs_count
                total_cost += 50 * symmetry_cost
            
            # 目标3：鼻尖点应该在目标位置
            nose_tip_idx = self.key_landmarks['nose_tip']
            if nose_tip_idx < len(transformed_vertices):
                nose_tip_error = np.linalg.norm(transformed_vertices[nose_tip_idx] - self.target_nose_tip) ** 2
                total_cost += 200 * nose_tip_error  # 最高权重
            
            # 目标4：几何中心应该接近目标中心
            current_center = np.mean(transformed_vertices, axis=0)
            center_error = np.linalg.norm(current_center - self.target_center) ** 2
            total_cost += 10 * center_error
            
            # 目标5：保持合理的缩放（防止模型变形过度）
            scale_penalty = (params[6] - 1.0) ** 2  # 缩放应该接近1
            total_cost += 5 * scale_penalty
            
            return total_cost
        
        # 初始参数：[tx, ty, tz, rx, ry, rz, scale]
        initial_params = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        
        # 参数边界
        bounds = [
            (-5, 5),    # tx
            (-5, 5),    # ty  
            (-5, 5),    # tz
            (-0.3, 0.3), # rx (旋转角度限制)
            (-0.3, 0.3), # ry
            (-0.3, 0.3), # rz
            (0.8, 1.2)   # scale (缩放限制)
        ]
        
        print("开始优化变换参数...")
        result = minimize(objective_function, initial_params, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            optimal_params = result.x
            print(f"优化成功！最终成本: {result.fun:.6f}")
            print(f"最优参数: 平移=({optimal_params[0]:.3f}, {optimal_params[1]:.3f}, {optimal_params[2]:.3f})")
            print(f"         旋转=({np.degrees(optimal_params[3]):.1f}°, {np.degrees(optimal_params[4]):.1f}°, {np.degrees(optimal_params[5]):.1f}°)")
            print(f"         缩放={optimal_params[6]:.3f}")
        else:
            print(f"优化失败: {result.message}")
            optimal_params = initial_params
        
        return create_transform_matrix(optimal_params)
    
    def apply_transform(self, rotation_matrix, translation):
        """应用变换到所有顶点"""
        print("应用变换到所有顶点...")
        
        # 应用旋转和平移
        self.vertices = (rotation_matrix @ self.vertices.T).T + translation
        
        # 强制调整中线点的X坐标为精确的0（微调）
        adjusted_count = 0
        for idx in self.centerline_indices:
            if idx < len(self.vertices):
                if abs(self.vertices[idx, 0]) < 0.05:  # 只调整接近中线的点
                    self.vertices[idx, 0] = 0.0
                    adjusted_count += 1
        
        print(f"微调了 {adjusted_count} 个中线点")
        
    def analyze_alignment_quality(self):
        """分析对齐质量"""
        print("\n=== 对齐质量分析 ===")
        
        # 检查中线点
        centerline_errors = []
        for idx in self.centerline_indices:
            if idx < len(self.vertices):
                x_coord = self.vertices[idx, 0]
                centerline_errors.append(abs(x_coord))
                if abs(x_coord) > 0.01:
                    print(f"中线点 {idx}: X = {x_coord:.6f}")
        
        avg_centerline_error = np.mean(centerline_errors) if centerline_errors else 0
        print(f"中线点平均X偏差: {avg_centerline_error:.6f}")
        
        # 检查对称性
        symmetry_errors = []
        for left_idx, right_idx in self.symmetric_pairs:
            if left_idx < len(self.vertices) and right_idx < len(self.vertices):
                left_point = self.vertices[left_idx]
                right_point = self.vertices[right_idx]
                
                x_error = abs(left_point[0] + right_point[0])
                y_error = abs(left_point[1] - right_point[1])
                z_error = abs(left_point[2] - right_point[2])
                
                total_error = x_error + y_error + z_error
                symmetry_errors.append(total_error)
        
        if symmetry_errors:
            avg_symmetry_error = np.mean(symmetry_errors)
            good_pairs = sum(1 for e in symmetry_errors if e < 0.01)
            print(f"对称点对平均误差: {avg_symmetry_error:.6f}")
            print(f"良好对称点对: {good_pairs}/{len(symmetry_errors)}")
        
        # 检查关键点位置
        nose_tip_idx = self.key_landmarks['nose_tip']
        if nose_tip_idx < len(self.vertices):
            nose_tip_pos = self.vertices[nose_tip_idx]
            nose_tip_error = np.linalg.norm(nose_tip_pos - self.target_nose_tip)
            print(f"鼻尖点位置: ({nose_tip_pos[0]:.6f}, {nose_tip_pos[1]:.6f}, {nose_tip_pos[2]:.6f})")
            print(f"鼻尖点误差: {nose_tip_error:.6f}")
        
        # 检查几何中心
        current_center = np.mean(self.vertices, axis=0)
        center_error = np.linalg.norm(current_center - self.target_center)
        print(f"当前几何中心: ({current_center[0]:.6f}, {current_center[1]:.6f}, {current_center[2]:.6f})")
        print(f"几何中心误差: {center_error:.6f}")
        
        # 综合评分
        overall_score = 100.0
        if avg_centerline_error > 0.01:
            overall_score -= 30
        if avg_symmetry_error > 0.01:
            overall_score -= 30
        if nose_tip_error > 0.1:
            overall_score -= 20
        if center_error > 0.5:
            overall_score -= 20
        
        print(f"\n对齐质量评分: {overall_score:.1f}/100")
        if overall_score >= 90:
            print("✅ 对齐质量优秀！")
        elif overall_score >= 70:
            print("✅ 对齐质量良好")
        else:
            print("⚠️ 对齐质量需要改进")
    
    def save_obj_file(self, output_path):
        """保存变换后的OBJ文件"""
        print(f"正在保存到: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as file:
            # 写入注释
            file.write("# 精确对齐后的面部模型\n")
            file.write("# 基于canonical_face_model.obj的准确分析进行对齐\n")
            
            # 写入顶点
            for vertex in self.vertices:
                file.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
                
            # 写入纹理坐标
            for vt in self.texture_coords:
                file.write(f"{vt}\n")
                
            # 写入面
            for face in self.faces:
                file.write(f"{face}\n")
                
        print(f"模型保存完成: {output_path}")
    
    def process_model(self, input_path, output_path=None):
        """处理模型的主函数"""
        print("=== 精确面部模型对齐工具 ===\n")
        
        # 加载模型
        self.load_obj_file(input_path)
        
        # 计算精确对齐变换
        rotation_matrix, translation = self.compute_precise_alignment_transform()
        
        # 应用变换
        self.apply_transform(rotation_matrix, translation)
        
        # 分析结果
        self.analyze_alignment_quality()
        
        # 保存结果
        if output_path is None:
            base_name = os.path.splitext(input_path)[0]
            output_path = f"{base_name}_precise_aligned.obj"
            
        self.save_obj_file(output_path)
        
        print(f"\n✅ 精确对齐处理完成!")
        print(f"原始文件: {input_path}")
        print(f"对齐文件: {output_path}")
        print("\n模型现在应该与canonical_face_model.obj具有相同的对齐方式。")
        
        return output_path

def main():
    """主函数"""
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # 查找最新的模型文件
        result_dir = "result_file"
        if not os.path.exists(result_dir):
            print("错误：未找到 result_file 目录")
            return
            
        model_files = [f for f in os.listdir(result_dir) if f.endswith('_face_model.obj') and 'aligned' not in f]
        if not model_files:
            print("错误：未找到面部模型文件")
            return
            
        # 按修改时间排序，选择最新的
        model_files.sort(key=lambda x: os.path.getmtime(os.path.join(result_dir, x)), reverse=True)
        input_file = os.path.join(result_dir, model_files[0])
        
    if not os.path.exists(input_file):
        print(f"错误：文件不存在 {input_file}")
        return
        
    # 创建工具实例并处理
    tool = PreciseFaceAlignmentTool()
    output_file = tool.process_model(input_file)
    
    print(f"\n处理完成！输出文件: {output_file}")

if __name__ == "__main__":
    main() 