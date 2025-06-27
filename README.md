**启动 3D 面具渲染器 python 版本为3.11**：
```bash
python fml/face_mask_renderer.py
```

# FaceMatrixLab

一个基于 MediaPipe 和 Open3D 的实时人脸跟踪与 3D 面具渲染系统，支持高精度人脸建模、表情驱动和 AR 效果。

## 🎯 项目概述

FaceMatrixLab 是一个完整的实时人脸分析与 3D 渲染解决方案，能够：

- 使用 MediaPipe 进行 468 个面部关键点的实时检测
- 基于 `facial_transformation_matrix` 实现精确的 3D 头部姿态跟踪
- 支持多种渲染模式：纹理贴图、Lambert 材质、线框显示
- 提供相机跟随功能，实现动态视角调整
- 支持数据记录与导出，便于后续分析

## ✨ 核心特性

### 🔍 人脸检测与跟踪
- **MediaPipe FaceLandmarker Tasks API**：468 个高精度 3D 面部关键点
- **姿态矩阵**：4×4 `facial_transformation_matrix` 提供完整的 TRS 变换
- **表情识别**：支持 ARKit BlendShapes（52 个表情权重）
- **实时性能**：30 FPS 稳定运行，低延迟响应

### 🎨 3D 渲染引擎
- **多渲染模式**：
  - 原始纹理复制（保持真实面部细节）
  - Lambert 漫反射材质（物理光照模型）
  - 双光源 MatCap 风格（ZBrush 风格渲染）
  - 线框模式（亚像素级细线支持）
- **透视投影**：真实 50mm 等效焦距相机模拟
- **动态光照**：环境光 + 主光源 + 补光系统

### 📷 相机系统
- **相机跟随**：3D 视角自动跟随人脸水平移动
- **透视校正**：支持真实相机内参校准
- **深度控制**：可调节基础深度和变化范围
- **宽高比修正**：针对 1280×720 分辨率优化

### 💾 数据记录
- **位置数据**：自动保存 landmarks 和相机位置
- **格式支持**：JSON 格式，包含时间戳和完整参数
- **批量导出**：支持 OBJ 模型实时导出

## 🏗️ 技术架构

```
FaceMatrixLab/
├── fml/                          # 核心模块
│   ├── face_mask_renderer.py     # 主渲染器（3D 面具 + AR）
│   ├── face_landmarker_camera.py # 人脸检测主控程序
│   ├── face_warper.py            # 核心渲染引擎
│   ├── precise_alignment_tool.py # 精确对齐工具
│   └── csv_to_obj_converter.py   # 数据格式转换
├── obj/                          # 3D 模型资源
│   ├── Andy_Wah_facemesh.obj     # 主要人脸模型
│   ├── canonical_face_model.obj  # MediaPipe 标准模型
│   └── enhanced_texture.png      # 高质量纹理贴图
├── Camera-Calibration/           # 相机校准工具
└── pos_param_rec/               # 数据记录目录
```

## 🚀 快速开始

### 环境要求

- Python 3.8+ (我使用的是3.11)
- Windows 10/11（推荐）
- 摄像头设备
- 4GB+ RAM

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本使用

1. **启动 3D 面具渲染器**：
```bash
python fml/face_mask_renderer.py
```

2. **相机校准**（可选）：
```bash
cd Camera-Calibration
python calibration_GUI.py
```

## 🎮 控制说明

### 面具渲染器控制键

| 按键 | 功能 |
|------|------|
| **B** | 切换摄像机背景显示 |
| **C** | 切换面具颜色 |
| **1-6** | 直接选择面具颜色 |
| **T** | 切换纹理贴图/统一颜色模式 |
| **L** | 切换原始 landmarks 显示 |
| **F** | 切换相机跟随功能 |
| **E** | 导出当前实时 3D 模型 |
| **Q** | 退出程序 |

## 📊 技术细节

### MediaPipe 数据流

系统处理三类核心数据：

1. **`face_landmarks`**（468×3）：
   - 归一化屏幕坐标 (x, y ∈ [0,1])
   - 相对深度值 (z)
   - 支持表情变化捕捉

2. **`facial_transformation_matrix`**（4×4）：
   ```
   [R11 R12 R13 Tx]    # 旋转矩阵 + X平移
   [R21 R22 R23 Ty]    # 旋转矩阵 + Y平移  
   [R31 R32 R33 Tz]    # 旋转矩阵 + Z平移
   [0   0   0   1 ]    # 齐次坐标
   ```
   - 单位：毫米（mm）
   - 提供完整的 3D 姿态信息

3. **`face_blendshapes`**（52 权重）：
   - ARKit 兼容的表情权重
   - 支持精细表情驱动

### 渲染管线

```
输入帧 → MediaPipe检测 → 坐标变换 → 三角剖分 → 逐三角形渲染 → 后处理 → 输出
```

**关键算法**：
- **Lambert 漫反射**：`I = Ia + Id × max(0, N·L)`
- **双光源系统**：主光源 + 正面补光 + Fresnel 边缘光
- **亚像素线框**：透明度模拟细线效果
- **边缘平滑**：双边滤波 + 距离变换

### 相机跟随算法

```python
# 计算人脸水平中心
face_center_x = (min_x + max_x) / 2.0

# 归一化偏移量
offset_normalized = (face_center_x - screen_center_x) / screen_center_x

# 平滑相机移动
target_offset = offset_normalized * sensitivity
current_offset = lerp(current_offset, target_offset, smoothing_factor)
```

## 🔧 配置参数

### 渲染参数

```python
# 基本设置
render_width = 1280          # 渲染宽度
render_height = 720          # 渲染高度
fps_target = 30              # 目标帧率

# 相机跟随
enable_camera_following = True    # 启用相机跟随
camera_follow_smoothing = 0.3     # 平滑系数 (0-1)
camera_sensitivity = 2.0          # 跟随灵敏度
camera_max_offset = 5.0           # 最大偏移限制

# 透视投影
perspective_base_depth = 45.0     # 基础深度 (cm)
perspective_depth_variation = 55.0 # 深度变化范围
focal_length_mm = 50.0            # 等效焦距 (mm)
```

### 材质参数

```python
# Lambert 光照
ambient_intensity = 0.7           # 环境光强度
main_diffuse_intensity = 0.8      # 主光源强度
front_diffuse_intensity = 0.4     # 补光强度

# 线框渲染
wireframe_thickness = 0.2         # 线框粗细
wireframe_alpha = 0.7             # 线框透明度
```

## 📁 数据格式

### Landmarks 数据结构

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "frame_count": 30,
  "screen_resolution": {
    "width": 1280,
    "height": 720
  },
  "original_landmarks": [
    {"x_000": 0.623, "y_000": 0.599, "z_000": -0.027},
    // ... 467 more points
  ],
  "pixel_landmarks": [
    {"x_000": 797.3, "y_000": 431.3, "z_000": -0.027},
    // ... 467 more points
  ],
  "camera_position": {
    "translation": {"x": 7.74, "y": -0.19, "z": -40.61},
    "rotation_matrix": [[0.995, -0.059, -0.083], ...],
    "intrinsic": {"fx": 1777.8, "fy": 1777.8, "cx": 640, "cy": 360}
  }
}
```

## 🎨 自定义模型

### 模型要求

1. **顶点顺序**：必须与 MediaPipe canonical face model 一致
2. **拓扑结构**：保持 468 个关键顶点的索引映射
3. **单位**：使用毫米（mm）作为单位
4. **UV 坐标**：如需纹理贴图，确保包含有效 UV

### 纹理贴图设置

**方法 1：修改 MTL 文件**
```mtl
newmtl face_mat
Ka 1 1 1
Kd 1 1 1
Ks 0 0 0
map_Kd enhanced_texture.png
```

**方法 2：代码中动态加载**
```python
tex_img = o3d.io.read_image("obj/enhanced_texture.png")
mesh.textures = [tex_img]
mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(mesh.triangles))
```

## 🐛 故障排除

### 常见问题

1. **模型不显示**
   - 检查 OBJ 文件路径
   - 确认模型包含有效顶点数据
   - 验证 UV 坐标范围 [0,1]

2. **相机跟随异常**
   - 调整 `camera_follow_smoothing` 参数
   - 检查 landmarks 检测质量
   - 重新校准相机内参

3. **纹理显示错误**
   - 确认纹理文件格式（PNG/JPG）
   - 检查 MTL 文件路径引用
   - 验证 triangle_material_ids 设置

4. **性能问题**
   - 降低渲染分辨率
   - 关闭不必要的可视化选项
   - 使用 GPU 加速（如可用）

### 调试模式

启用调试输出：
```python
debug_mode = True
show_fps = True
show_landmarks = True
```

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发环境设置

1. Fork 本仓库
2. 创建功能分支：`git checkout -b feature/new-feature`
3. 提交更改：`git commit -am 'Add new feature'`
4. 推送分支：`git push origin feature/new-feature`
5. 创建 Pull Request

### 代码规范

- 使用 Python PEP 8 标准
- 添加适当的注释和文档字符串
- 保持函数功能单一且清晰
- 提交前运行测试用例

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [MediaPipe](https://mediapipe.dev/) - Google 开源的机器学习框架
- [Open3D](http://www.open3d.org/) - 3D 数据处理库
- [OpenCV](https://opencv.org/) - 计算机视觉库

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 GitHub Issue
- 发送邮件至项目维护者
- 参与社区讨论

---

**FaceMatrixLab** - 让人脸跟踪与 3D 渲染变得简单而强大！ 