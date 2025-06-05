#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV Landmarks 转 OBJ 文件转换器
将人脸landmarks的CSV数据转换为标准的OBJ 3D模型文件
"""

import csv
import os
import sys


class LandmarksToObjConverter:
    def __init__(self, template_obj_path='obj/canonical_face_model.obj'):
        """初始化转换器"""
        self.template_obj_path = template_obj_path
        self.texture_coords = []
        self.faces = []
        self.load_template()
    
    def load_template(self):
        """从模板OBJ文件中加载纹理坐标和面信息"""
        if not os.path.exists(self.template_obj_path):
            print(f"模板文件不存在: {self.template_obj_path}")
            print("请确保 canonical_face_model.obj 文件存在于 obj/ 目录中")
            return False
        
        print(f"正在加载模板文件: {self.template_obj_path}")
        
        try:
            with open(self.template_obj_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('vt '):  # 纹理坐标
                        parts = line.split()
                        if len(parts) >= 3:
                            u, v = float(parts[1]), float(parts[2])
                            self.texture_coords.append((u, v))
                    elif line.startswith('f '):  # 面信息
                        self.faces.append(line)
            
            print(f"加载完成: {len(self.texture_coords)} 个纹理坐标, {len(self.faces)} 个面")
            return True
            
        except Exception as e:
            print(f"加载模板文件失败: {e}")
            return False
    
    def load_landmarks_csv(self, csv_path):
        """从CSV文件加载landmarks数据"""
        landmarks = []
        
        if not os.path.exists(csv_path):
            print(f"CSV文件不存在: {csv_path}")
            return None
        
        print(f"正在加载CSV文件: {csv_path}")
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    point_id = int(row['point_id'])
                    x = float(row['x'])
                    y = float(row['y'])
                    z = float(row['z'])
                    landmarks.append((point_id, x, y, z))
            
            # 按point_id排序确保顺序正确
            landmarks.sort(key=lambda x: x[0])
            
            print(f"加载了 {len(landmarks)} 个landmarks点")
            return landmarks
            
        except Exception as e:
            print(f"加载CSV文件失败: {e}")
            return None
    
    def normalize_landmarks(self, landmarks):
        """标准化landmarks坐标"""
        print("正在标准化landmarks坐标...")
        
        # MediaPipe的landmarks是归一化的 (0-1范围)
        # 需要转换为3D坐标系统，匹配canonical_face_model.obj的尺寸
        
        normalized_landmarks = []
        for point_id, x, y, z in landmarks:
            # 坐标转换包括：
            # 1. 16:9宽高比修正
            # 2. Z轴正方向朝前
            # 3. 全部XYZ总体缩放55倍
            
            # 处理16:9宽高比问题，X轴需要乘以16/9来恢复原始比例
            aspect_ratio_correction = 16.0 / 9.0  # ≈ 1.777
            
            # 总体缩放因子
            overall_scale = 55.0
            
            # 转换坐标系：MediaPipe使用左上角为原点，Y向下
            # 3D模型使用中心为原点，Y向上，Z向前
            x_3d = (x - 0.5) * aspect_ratio_correction * overall_scale  # 修正宽高比并缩放55倍
            y_3d = -(y - 0.5) * overall_scale # 翻转Y轴并缩放55倍
            z_3d = -z * aspect_ratio_correction * overall_scale  # Z轴正方向朝前并缩放55倍
            
            normalized_landmarks.append((x_3d, y_3d, z_3d))
        
        return normalized_landmarks
    
    def save_obj(self, landmarks, output_path):
        """保存为OBJ文件"""
        print(f"正在保存OBJ文件: {output_path}")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # 写入文件头注释
                f.write("# OBJ file generated from MediaPipe face landmarks\n")
                f.write(f"# Original landmarks count: {len(landmarks)}\n")
                f.write(f"# Texture coordinates: {len(self.texture_coords)}\n")
                f.write(f"# Faces: {len(self.faces)}\n\n")
                
                # 写入顶点坐标 (v)
                for i, (x, y, z) in enumerate(landmarks):
                    f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
                
                f.write("\n")
                
                # 写入纹理坐标 (vt)
                for u, v in self.texture_coords:
                    f.write(f"vt {u:.6f} {v:.6f}\n")
                
                f.write("\n")
                
                # 写入面信息 (f)
                for face_line in self.faces:
                    f.write(f"{face_line}\n")
            
            print(f"OBJ文件保存成功: {output_path}")
            return True
            
        except Exception as e:
            print(f"保存OBJ文件失败: {e}")
            return False
    
    def convert(self, csv_path, output_path=None):
        """执行转换"""
        print("=" * 60)
        print("开始CSV到OBJ转换")
        print("=" * 60)
        
        # 加载landmarks数据
        landmarks_data = self.load_landmarks_csv(csv_path)
        if landmarks_data is None:
            return False
        
        # 检查landmarks数量
        if len(landmarks_data) != 468:
            print(f"警告: landmarks数量不正确 ({len(landmarks_data)}/468)")
            print("MediaPipe人脸模型应该有468个landmarks点")
        
        # 标准化坐标
        normalized_landmarks = self.normalize_landmarks(landmarks_data)
        
        # 创建输出文件夹
        output_dir = "result_file"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建输出文件夹: {output_dir}")
        
        # 生成输出文件名
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(csv_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_face_model.obj")
        else:
            # 如果用户指定了输出路径，也放到result_file文件夹中
            output_filename = os.path.basename(output_path)
            output_path = os.path.join(output_dir, output_filename)
        
        # 保存OBJ文件
        success = self.save_obj(normalized_landmarks, output_path)
        
        if success:
            print("=" * 60)
            print("转换完成！")
            print(f"输入文件: {csv_path}")
            print(f"输出文件: {output_path}")
            print("=" * 60)
        
        return success


def main():
    """主函数"""
    print("MediaPipe Landmarks CSV 转 OBJ 文件转换器")
    print("=" * 60)
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python csv_to_obj_converter.py <CSV文件路径> [输出OBJ文件路径]")
        print("\n示例:")
        print("  python csv_to_obj_converter.py face_landmarks_1234567890.csv")
        print("  python csv_to_obj_converter.py face_landmarks_1234567890.csv my_face_model.obj")
        return
    
    csv_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # 创建转换器
    converter = LandmarksToObjConverter()
    
    # 执行转换
    converter.convert(csv_path, output_path)


if __name__ == "__main__":
    main() 