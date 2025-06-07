#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机标定脚本 - 使用棋盘格图片进行相机内参标定
"""

import cv2
import glob
import numpy as np
import os

def calibrate_camera():
    """使用棋盘格图片进行相机标定"""
    
    # 尝试多种棋盘格配置
    chessboard_configs = [
        (6, 9),   # 6×9
        (9, 6),   # 9×6 
        (7, 10),  # 7×10
        (10, 7),  # 10×7
        (8, 11),  # 8×11
        (11, 8),  # 11×8
        (5, 8),   # 5×8
        (8, 5),   # 8×5
        (4, 7),   # 4×7
        (7, 4),   # 7×4
    ]
    
    square_size = 25.0  # 棋盘格方格边长（毫米）
    
    print("=== 相机标定程序 ===")
    print("正在自动检测棋盘格配置...")
    
    # 读取标定图片
    images = glob.glob('calib/*.jpg')
    if not images:
        print("❌ 错误：在calib/文件夹中未找到jpg图片")
        print("请确保calib/文件夹中有棋盘格标定图片")
        return False
    
    print(f"📸 找到 {len(images)} 张标定图片")
    
    # 显示第一张图片让用户确认
    first_img = cv2.imread(images[0])
    if first_img is not None:
        print(f"📷 显示第一张图片: {os.path.basename(images[0])}")
        cv2.imshow('第一张标定图片 - 按任意键继续', first_img)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
    
    # 尝试不同的棋盘格配置
    best_config = None
    best_count = 0
    
    for rows, cols in chessboard_configs:
        print(f"\n🔍 尝试棋盘格配置: {rows}×{cols}")
        
        # 准备3D物体点坐标
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
        
        successful_count = 0
        
        # 测试前5张图片
        for i, fname in enumerate(images[:5]):
            img = cv2.imread(fname)
            if img is None:
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
            
            if ret:
                successful_count += 1
        
        print(f"  前5张图片中成功检测: {successful_count}/5")
        
        if successful_count > best_count:
            best_count = successful_count
            best_config = (rows, cols)
    
    if best_config is None or best_count == 0:
        print("\n❌ 无法找到合适的棋盘格配置")
        print("请检查：")
        print("1. 图片中是否有清晰的棋盘格")
        print("2. 棋盘格是否完整可见")
        print("3. 光照是否充足，对比度是否足够")
        return False
    
    rows, cols = best_config
    print(f"\n✅ 最佳棋盘格配置: {rows}×{cols} (成功率: {best_count}/5)")
    print(f"方格大小: {square_size} mm")
    
    # 使用最佳配置进行完整标定
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    
    # 存储所有图像的物体点和图像点
    objpoints = []  # 3D点
    imgpoints = []  # 2D点
    
    # 处理所有图片
    successful_images = 0
    image_shape = None
    
    for i, fname in enumerate(images):
        print(f"处理第 {i+1}/{len(images)} 张图片: {os.path.basename(fname)}")
        
        img = cv2.imread(fname)
        if img is None:
            print(f"⚠️ 无法读取图片: {fname}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 保存图像尺寸
        if image_shape is None:
            image_shape = gray.shape[::-1]  # (width, height)
        
        # 寻找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        
        if ret:
            print(f"  ✅ 成功检测到角点")
            
            # 精确化角点位置
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # 保存物体点和图像点
            objpoints.append(objp)
            imgpoints.append(corners2)
            successful_images += 1
            
            # 绘制角点（可选，用于验证）
            img_with_corners = img.copy()
            cv2.drawChessboardCorners(img_with_corners, (cols, rows), corners2, ret)
            
            # 显示检测结果
            cv2.imshow('棋盘格角点检测', img_with_corners)
            cv2.waitKey(200)  # 自动继续
            
        else:
            print(f"  ❌ 未检测到角点")
    
    cv2.destroyAllWindows()
    
    if successful_images < 10:
        print(f"❌ 警告：只有 {successful_images} 张图片成功检测到角点")
        print("建议至少有10张成功的标定图片以获得更好的标定效果")
        if successful_images < 3:
            print("标定图片数量太少，无法进行标定")
            return False
    
    print(f"📊 成功处理 {successful_images} 张图片，开始标定...")
    
    # 执行相机标定
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_shape, None, None
    )
    
    if not ret:
        print("❌ 相机标定失败")
        return False
    
    # 计算标定误差
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                         camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    
    mean_error = total_error / len(objpoints)
    
    # 保存标定结果
    calibration_file = 'calib.npz'
    np.savez(calibration_file, 
             K=camera_matrix, 
             dist=dist_coeffs,
             image_shape=image_shape,
             mean_error=mean_error)
    
    # 显示标定结果
    print("\n=== 标定结果 ===")
    print("✅ 相机标定成功！")
    print(f"📁 标定文件已保存: {calibration_file}")
    print(f"📐 图像尺寸: {image_shape[0]}×{image_shape[1]}")
    print(f"📊 平均重投影误差: {mean_error:.3f} 像素")
    print("\n🎯 相机内参矩阵 (K):")
    print(camera_matrix)
    print(f"\n📏 焦距: fx={camera_matrix[0,0]:.2f}, fy={camera_matrix[1,1]:.2f}")
    print(f"📍 主点: cx={camera_matrix[0,2]:.2f}, cy={camera_matrix[1,2]:.2f}")
    print(f"\n🔧 畸变系数: {dist_coeffs.ravel()}")
    
    if mean_error < 1.0:
        print("🎉 标定质量：优秀 (误差 < 1.0 像素)")
    elif mean_error < 2.0:
        print("✅ 标定质量：良好 (误差 < 2.0 像素)")
    else:
        print("⚠️ 标定质量：一般 (误差 > 2.0 像素，建议重新拍摄更多高质量图片)")
    
    return True

if __name__ == "__main__":
    success = calibrate_camera()
    if success:
        print("\n🎯 标定完成！现在可以运行 face_matrix_lab_render.py 使用精确的3D跟踪")
    else:
        print("\n❌ 标定失败，请检查标定图片质量")
    
    input("\n按回车键退出...") 