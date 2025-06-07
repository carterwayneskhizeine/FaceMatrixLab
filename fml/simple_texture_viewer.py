import open3d as o3d
import numpy as np
import os

def view_textured_face_model():
    """
    简单的纹理面部模型查看器
    """
    # 文件路径
    model_path = "obj/Andy_Wah_facemesh.obj"
    texture_path = "enhanced_texture.png"
    
    print("=== 简单纹理查看器 ===")
    print(f"加载模型: {model_path}")
    print(f"加载纹理: {texture_path}")
    
    # 加载模型
    mesh = o3d.io.read_triangle_mesh(model_path)
    print(f"模型信息: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角形")
    print(f"UV坐标: {'有' if mesh.has_triangle_uvs() else '无'}")
    
    # 加载纹理
    if os.path.exists(texture_path):
        texture = o3d.io.read_image(texture_path)
        texture_array = np.asarray(texture)
        print(f"纹理信息: {texture_array.shape} 尺寸")
        
        # 将纹理应用到模型
        mesh.textures = [texture]
        print("纹理已应用到模型")
    else:
        print("未找到纹理文件，使用默认颜色")
        mesh.paint_uniform_color([0.7, 0.7, 0.7])  # 灰色
    
    # 计算法向量（如果没有的话）
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
        print("已计算顶点法向量")
    
    # 显示模型
    print("启动3D查看器...")
    print("使用说明:")
    print("- 鼠标左键拖拽: 旋转模型")
    print("- 鼠标滚轮: 缩放")
    print("- 鼠标右键拖拽: 平移")
    print("- 按 'H' 键: 显示帮助")
    print("- 按 'Q' 键或关闭窗口: 退出")
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Andy Wah 面部模型 - 纹理查看器", 
                      width=1200, height=900)
    
    # 添加模型到场景
    vis.add_geometry(mesh)
    
    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.show_coordinate_frame = True  # 显示坐标轴
    render_option.background_color = np.asarray([0.1, 0.1, 0.1])  # 深灰色背景
    
    # 运行可视化
    vis.run()
    
    # 可选：保存截图
    try:
        print("保存截图...")
        vis.capture_screen_image("textured_face_model_screenshot.png")
        print("截图已保存为: textured_face_model_screenshot.png")
    except:
        print("截图保存失败")
    
    # 清理资源
    vis.destroy_window()
    
    print("查看器已关闭")

def compare_with_without_texture():
    """
    对比有纹理和无纹理的模型
    """
    model_path = "obj/Andy_Wah_facemesh.obj"
    texture_path = "enhanced_texture.png"
    
    # 加载模型
    mesh1 = o3d.io.read_triangle_mesh(model_path)  # 带纹理
    mesh2 = o3d.io.read_triangle_mesh(model_path)  # 不带纹理
    
    # 设置纹理
    if os.path.exists(texture_path):
        texture = o3d.io.read_image(texture_path)
        mesh1.textures = [texture]
    
    # 给第二个模型设置单色
    mesh2.paint_uniform_color([0.8, 0.6, 0.4])  # 肌肤色
    
    # 计算法向量
    mesh1.compute_vertex_normals()
    mesh2.compute_vertex_normals()
    
    # 分离两个模型的位置
    mesh2.translate([0.3, 0, 0])  # 向右移动第二个模型
    
    print("显示对比: 左侧为带纹理模型，右侧为单色模型")
    
    # 显示两个模型
    o3d.visualization.draw_geometries([mesh1, mesh2],
                                      window_name="纹理对比 - 左:纹理模型 右:单色模型",
                                      width=1400, height=800)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare_with_without_texture()
    else:
        view_textured_face_model() 