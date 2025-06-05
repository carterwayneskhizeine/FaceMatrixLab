#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基本功能测试
"""

import os
import numpy as np

def test_open3d():
    """测试Open3D"""
    try:
        import open3d as o3d
        print("✅ Open3D导入成功")
        print(f"   版本: {o3d.__version__}")
        
        # 测试基本几何体创建
        mesh = o3d.geometry.TriangleMesh.create_box()
        print("✅ Open3D基本几何体创建成功")
        
        return True
    except Exception as e:
        print(f"❌ Open3D测试失败: {e}")
        return False

def test_mediapipe():
    """测试MediaPipe"""
    try:
        import mediapipe as mp
        print("✅ MediaPipe导入成功")
        print(f"   版本: {mp.__version__}")
        
        # 测试基本模块
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        print("✅ MediaPipe FaceLandmarker模块导入成功")
        
        return True
    except Exception as e:
        print(f"❌ MediaPipe测试失败: {e}")
        return False

def test_model_loading():
    """测试模型加载"""
    try:
        import open3d as o3d
        
        model_path = "../obj/Andy_Wah_facemesh.obj"
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return False
        
        mesh = o3d.io.read_triangle_mesh(model_path)
        vertices = np.asarray(mesh.vertices)
        
        print(f"✅ 模型加载成功:")
        print(f"   文件: {model_path}")
        print(f"   顶点数: {len(vertices)}")
        print(f"   面数: {len(mesh.triangles)}")
        print(f"   坐标范围:")
        print(f"     X: [{vertices[:, 0].min():.2f}, {vertices[:, 0].max():.2f}]")
        print(f"     Y: [{vertices[:, 1].min():.2f}, {vertices[:, 1].max():.2f}]")
        print(f"     Z: [{vertices[:, 2].min():.2f}, {vertices[:, 2].max():.2f}]")
        
        return True
    except Exception as e:
        print(f"❌ 模型加载测试失败: {e}")
        return False

def test_opencv():
    """测试OpenCV摄像头"""
    try:
        import cv2
        print("✅ OpenCV导入成功")
        print(f"   版本: {cv2.__version__}")
        
        # 简单测试摄像头
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ 摄像头访问成功")
            cap.release()
            return True
        else:
            print("⚠️ 摄像头无法打开")
            return False
            
    except Exception as e:
        print(f"❌ OpenCV测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=== FaceMatrixLab 基本功能测试 ===\n")
    
    tests = [
        ("Open3D", test_open3d),
        ("MediaPipe", test_mediapipe),
        ("模型加载", test_model_loading),
        ("OpenCV摄像头", test_opencv),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"🧪 测试 {test_name}:")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results[test_name] = False
        print()
    
    # 总结
    print("=" * 50)
    print("测试结果总结:")
    passed = 0
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n通过率: {passed}/{len(tests)} ({passed/len(tests)*100:.1f}%)")
    
    if passed == len(tests):
        print("🎉 所有测试通过！可以运行完整渲染器。")
    else:
        print("⚠️ 部分测试失败，请先解决相关问题。")

if __name__ == "__main__":
    main() 