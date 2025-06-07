import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import sys
import os
import numpy as np

def apply_texture_to_face_model():
    """
    给Andy_Wah_facemesh.obj模型应用enhanced_texture.png颜色贴图
    """
    # 定义模型和纹理文件路径
    model_path = "obj/Andy_Wah_facemesh.obj"
    texture_path = "enhanced_texture.png"
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        return
    
    if not os.path.exists(texture_path):
        print(f"错误：找不到纹理文件 {texture_path}")
        return
    
    print(f"正在加载模型: {model_path}")
    print(f"正在加载纹理: {texture_path}")
    
    # 加载三角网格模型
    mesh = o3d.io.read_triangle_mesh(model_path)
    
    if len(mesh.vertices) == 0:
        print("错误：模型加载失败或模型为空")
        return
    
    print(f"模型加载成功，顶点数: {len(mesh.vertices)}, 三角形数: {len(mesh.triangles)}")
    
    # 确保模型有法向量（用于光照计算）
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
        print("已计算顶点法向量")
    
    # 加载纹理图像
    try:
        texture_image = o3d.io.read_image(texture_path)
        print("纹理贴图加载成功")
        print(f"纹理尺寸: {np.asarray(texture_image).shape}")
    except Exception as e:
        print(f"纹理贴图加载失败: {e}")
        return
    
    # 创建材质记录
    material = rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.albedo_img = texture_image
    material.base_color = [1.0, 1.0, 1.0, 1.0]  # 白色基础色
    material.base_metallic = 0.0  # 非金属
    material.base_roughness = 0.5  # 中等粗糙度
    
    # 使用Open3D的可视化功能显示带纹理的模型
    print("开始可视化...")
    print("提示：在可视化窗口中可以使用鼠标旋转、缩放模型")
    print("按 'Q' 键退出可视化")
    
    try:
        o3d.visualization.draw([{
            "name": "textured_face_model", 
            "geometry": mesh, 
            "material": material
        }])
    except Exception as e:
        print(f"可视化失败: {e}")
        print("尝试使用传统可视化方法...")
        # 如果高级可视化失败，使用传统方法
        o3d.visualization.draw_geometries([mesh], 
                                          window_name="Andy Wah Face Model",
                                          width=1024, height=768)

def show_model_info():
    """
    显示模型信息
    """
    model_path = "obj/Andy_Wah_facemesh.obj"
    texture_path = "enhanced_texture.png"
    
    print("=== 模型信息 ===")
    if os.path.exists(model_path):
        mesh = o3d.io.read_triangle_mesh(model_path)
        print(f"模型文件: {model_path}")
        print(f"顶点数: {len(mesh.vertices)}")
        print(f"三角形数: {len(mesh.triangles)}")
        print(f"是否有纹理坐标: {mesh.has_triangle_uvs()}")
        print(f"是否有顶点法向量: {mesh.has_vertex_normals()}")
    
    if os.path.exists(texture_path):
        texture = o3d.io.read_image(texture_path)
        texture_array = np.asarray(texture)
        print(f"纹理文件: {texture_path}")
        print(f"纹理尺寸: {texture_array.shape}")
        print(f"纹理类型: {texture_array.dtype}")

def main():
    """
    主函数
    """
    print("=== Open3D 纹理贴图应用程序 ===")
    
    # 显示模型和纹理信息
    show_model_info()
    print()
    
    print("正在为Andy_Wah_facemesh.obj应用enhanced_texture.png纹理...")
    apply_texture_to_face_model()
    
    print("程序结束")

if __name__ == "__main__":
    main() 