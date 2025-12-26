# Device Number Reading - 设备数码显示读数分析系统

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)
![License](https://img.shields.io/badge/license-MIT-orange.svg)

**基于计算机视觉的 7 段数码管自动识别与数据分析工具**

[English](#english) | [中文](#chinese)

</div>

---

<a name="chinese"></a>

## 📖 项目简介

Device Number Reading 是一个专为工业设备数码显示读数设计的图像识别分析系统（v5.0.0）。该系统使用基于 PaddleOCR SVTR_Tiny 深度学习模型的计算机视觉技术，能够自动识别 7 段/8 段数码管显示的数字，并将时间序列数据导出为 Excel 格式，适用于实验监测、设备读数记录等场景。采用模块化架构设计，代码结构清晰，易于维护和扩展。

### ✨ 核心特性

- 🎯 **高精度识别**：基于 SVTR_Tiny 数码管专用深度学习模型
- 🚀 **专用优化**：针对数码管场景训练，识别准确率更高
- 🔧 **灵活配置**：支持多种预处理选项和参数调节
- 📦 **轻量模型**：模型大小仅 24MB，推理速度更快
- 🔢 **小数点位置调整**：支持识别后调整小数点位置和精度
- 📊 **智能数据可视化**：实时绘制读数散点图，根据置信度动态标注颜色
- 🎨 **置信度可视化**：不同置信度用不同颜色表示（绿色=高，橙色=中，红色=低，黑色=失败）
- 🖱️ **交互式曲线图**：点击数据点即可跳转到对应图像帧
- ✏️ **手动校正功能**：支持对识别结果进行手动修正（Ctrl+R 或双击图像）
- 💾 **数据导出**：一键导出 Excel 格式数据（含置信度信息）
- 🎓 **导出训练数据**：支持将识别结果导出为 PaddleOCR 训练数据格式（Ctrl+T）
- 🖼️ **ROI 选择**：可视化感兴趣区域选择
- 🚀 **批量处理**：支持图像序列批量识别

### 🎥 应用场景

- 工业设备数码表读数自动记录
- 实验室仪器数据采集
- 历史设备监控数据提取
- 时间序列数据分析

---

## 🚀 快速开始

### 环境要求

- **操作系统**：Windows 10/11, Linux, macOS
- **Python 版本**：3.8 或更高
- **内存**：建议 4GB 以上

### 安装步骤

1. **克隆或下载项目**

```bash
git clone https://github.com/yourusername/Device_Number_Reading.git
cd Device_Number_Reading
```

2. **创建虚拟环境（推荐）**

```bash
# 使用 conda
conda create -n device_reading python=3.9
conda activate device_reading

# 或使用 venv
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

3. **安装依赖包**

```bash
pip install -r requirements.txt

# 注意：本项目使用预训练的 SVTR_Tiny 数码管专用模型（已包含在项目中）
# 模型文件位于 ./svtr_tiny_digital/ 目录（约 24MB）
# 推理模型（推荐，速度更快）：
#   - inference.pdmodel（模型结构）
#   - inference.pdiparams（模型权重）
# 训练模型（备用）：
#   - best_accuracy.pdparams（模型权重）
# 程序自动优先使用推理模型，如不存在则使用训练模型
# 无需额外下载，开箱即用
```

4. **运行程序**

```bash
python Device_Reading_Analyzer.py
```

---

## 📘 使用指南

### 基本工作流程

```
加载图像序列 → 选择ROI → 设置参数 → 处理图像 → 查看/校正 → 导出数据
```

### 详细步骤

#### 1️⃣ 加载图像序列

- 点击 **"Load Image Sequence"** 按钮
- 选择包含设备读数图像的文件夹
- 支持格式：`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`

#### 2️⃣ 选择感兴趣区域 (ROI)

- 点击 **"Select ROI"** 按钮
- 在图像上拖动鼠标框选数码显示区域
- 确保 ROI 完整包含所有数字和小数点
- 可使用 **"Clear ROI"** 重新选择

#### 3️⃣ 参数配置

**时间参数**

- **Time Interval**：图像之间的时间间隔
- **Time Unit**：时间单位（milliseconds, seconds, minutes, hours）
- **Start Time**：起始时间值

**识别参数**

- **Reading Unit**：读数单位（可选，用于标注）
- **Decimal Position**：小数点位置调整
  - **Keep**：保持原始识别结果（默认）
  - **None**：移除小数点，得到整数
  - **0.1**：调整为 1 位小数精度
  - **0.01**：调整为 2 位小数精度
  - **0.001**：调整为 3 位小数精度
  - **0.0001**：调整为 4 位小数精度
  - 用途：当数码管显示不包含小数点，但实际读数需要小数位时使用

**预处理参数**

- **Erosion Size**：腐蚀运算核大小（0 表示禁用，>0 表示启用，使用 disk 形状核）
  - 启用后会自动检测并处理小数点
  - 将小数点替换为自适应大小的实心圆点
- **Closing Size**：闭运算核大小（0 表示禁用，>0 表示启用，使用矩形核）
  - 自动保护小数点，不对其进行闭运算
- **Preview Preprocessing**：预览预处理效果

#### 4️⃣ 处理图像

- 点击 **"Process Images"** 开始批量识别
- 实时查看：
  - 当前处理进度
  - 识别结果日志
  - 实时更新的读数散点图（根据置信度显示不同颜色）
    - 🟢 绿色：高置信度 (≥ 0.9)
    - 🟠 橙色：中等置信度 (0.75 ~ 0.89)
    - 🔴 红色：低置信度 (< 0.75)
    - ▲ 黑色三角形：识别失败 (NaN)
- 可随时点击 **"Stop Processing"** 中止处理

#### 5️⃣ 查看和交互

**智能曲线图功能**

- 处理完成后，散点图会显示完整的读数数据
- **置信度颜色编码**：每个数据点的颜色反映识别置信度
  - 🟢 **绿色**：高置信度 (≥ 0.9) - 识别结果高度可信
  - 🟠 **橙色**：中等置信度 ([0.75, 0.9)) - 识别结果基本可信，建议人工复核
  - 🔴 **红色**：低置信度 (< 0.75) - 识别结果可信度低，需要人工复核
  - ▲ **黑色三角形**：识别失败 (NaN) - 未能识别，需要检查原图
- **交互式跳转**：
  - 点击散点图上的任意数据点
  - 图像预览区域会自动跳转到对应的图像帧
  - 日志窗口显示跳转信息（帧号、时间、读数、置信度）
  - 注意：图像处理期间点击跳转功能被禁用

#### 5️⃣.5 手动校正（新功能）

当批量处理完成后，如果发现个别识别结果有误，可进行手动校正：

**触发方式**：

- **快捷键**：导航到需要校正的图像后，按 `Ctrl+R`
- **双击**：直接双击预览窗口中的图像

**校正流程**：

1. 在弹出的对话框中输入正确的数值
2. 点击 `Confirm` 或按 `Enter` 确认
3. 点击 `Cancel` 或按 `Esc` 取消

**校正效果**：

- 散点图自动更新对应数据点
- 预览窗口显示更新后的标注
- 置信度自动设为 1.0（100%）
- 导出 Excel 时使用校正后的值

#### 6️⃣ 导出数据

- 处理完成后，点击 **"Export to Excel"**
- 选择保存位置和文件名
- 数据包含：时间列、读数列、置信度列

#### 7️⃣ 导出训练数据（新功能）

将识别结果导出为 PaddleOCR 训练数据格式，便于后续模型微调：

**触发方式**：

- 点击 **File → Export to Train Data**
- 或按 **Ctrl+T** 快捷键

**导出内容**：

- **图像文件**：ROI 区域裁剪后的原始图像（不含预处理）
- **标签文件**：`rec_gt_train.txt`（Tab 分隔格式）

**导出目录结构**：

```
导出目录/
├── train_images/           # 图像目录
│   ├── img_00000.png
│   ├── img_00001.png
│   └── ...
└── rec_gt_train.txt        # 标签文件
```

**标签文件格式**：

```
train_images/img_00000.png	-70.00
train_images/img_00001.png	25.30
```

**注意事项**：

- 仅在批量处理完成后可用
- 无效识别结果会被自动跳过
- 手动校正后的结果会被正确导出

---

## 🔧 技术架构

### 核心技术栈

| 技术             | 版本     | 用途           |
| ---------------- | -------- | -------------- |
| **Python**       | 3.8+     | 主要开发语言   |
| **OpenCV**       | 4.5+     | 图像处理       |
| **PaddleOCR**    | 2.9.1    | OCR 推理引擎   |
| **PaddlePaddle** | 2.6.2    | 深度学习框架   |
| **SVTR_Tiny**    | Custom   | 数码管识别模型 |
| **NumPy**        | 1.24+    | 数值计算       |
| **Tkinter**      | Built-in | GUI 界面       |
| **Matplotlib**   | 3.3+     | 数据可视化     |
| **Pandas**       | 1.3+     | 数据处理       |

### 识别算法

#### SVTR_Tiny 数码管专用识别

```
图像预处理 → 组件间距扩展 → SVTR_Tiny识别 → 结果验证
```

**优势**：

- 🎯 **专用训练**：针对数码管场景专门训练的模型
- 📦 **轻量高效**：模型仅 24MB，推理速度快
- 🎓 **高准确率**：专用字典（仅 0-9、-、.），减少误识别
- 🔧 **开箱即用**：模型已包含在项目中，无需下载
- 🚀 **SVTR 架构**：先进的文本识别架构，识别准确率更高
- ⚡ **推理优化**：优先使用推理模型（inference.pdmodel + inference.pdiparams），速度更快、部署优化

**模型架构**：

- **算法**：SVTR (Scene Text Recognition with a Single Visual Model)
- **骨干网络**：SVTRNet (6-layer Local + 6-layer Global)
- **解码器**：CTC (Connectionist Temporal Classification)
- **空间变换**：STN (Spatial Transformer Network) 透视校正
- **输入尺寸**：3 × 64 × 256 (C × H × W)
- **自定义字典**：仅包含 0-9、-、. 共 12 个字符

**技术特点**：

- 字符白名单：仅识别"-0123456789."
- 置信度过滤机制
- 自动图像预处理优化
- 支持 GPU 加速推理
- 推理模型优先：自动检测并使用推理模型（inference.pdmodel + inference.pdiparams），如不存在则使用训练模型（best_accuracy.pdparams）

### 图像预处理流程

```python
原始图像
  ↓
ROI提取
  ↓
灰度化 (cvtColor)
  ↓
OTSU自适应二值化 (threshold)
  ↓
小对象过滤 (connectedComponents)
  ↓
边界添加 (copyMakeBorder, 15像素)
  ↓
可选：腐蚀运算 (erosion, disk形状核)
  ↓
  ├─ 如果启用腐蚀且检测到小数点：
  │   ├─ 检测小数点（面积最小且<倒数第二小的1/4）
  │   ├─ 扩展其他对象与小数点的水平间距（4倍腐蚀核大小）
  │   ├─ 对其他对象进行膨胀（恢复原状）
  │   └─ 将小数点替换为实心圆点（直径=max(4×腐蚀核, ROI高度/15)）
  ↓
可选：闭运算 (closing, 矩形核，保护小数点)
  ↓
尺寸调整 (resize, 确保高度≥32像素)
  ↓
PaddleOCR识别
```

### 项目结构

```
Device_Number_Reading/
├── Device_Reading_Analyzer.py    # 主程序入口（模块化架构，v5.0.0）
├── requirements.txt               # 依赖包列表
├── config.json                    # 配置文件（可选）
├── digital_dict.txt               # 自定义字典（0-9, -, .）
├── README.md                      # 项目文档
├── QUICKSTART.md                  # 快速开始指南
├── PROJECT_STRUCTURE.md           # 项目结构说明
├── LICENSE                        # MIT许可证
│
├── src/                           # 源代码模块目录
│   ├── __init__.py                # 包初始化
│   ├── main_window.py             # 主窗口模块（完整实现，~1700行）
│   │
│   ├── core/                      # 核心功能模块
│   │   ├── __init__.py
│   │   ├── image_processor.py     # 图像处理（完整实现，~570行）
│   │   └── digit_recognizer.py    # 数字识别（完整实现，~535行）
│   │
│   ├── utils/                     # 工具模块（预留）
│   │   └── __init__.py
│   │
│   └── resources/                 # 资源文件
│       ├── __init__.py
│       ├── help_content_cn.txt    # 中文使用指南（~550行）
│       └── help_content_en.txt    # 英文使用指南（~550行）
│
├── svtr_tiny_digital/             # SVTR_Tiny 模型目录
│   ├── inference.pdmodel          # 推理模型结构（推荐）
│   ├── inference.pdiparams        # 推理模型权重（推荐，~24MB）
│   ├── best_accuracy.pdparams     # 训练模型权重（备用，~24MB）
│   ├── config.yml                 # 模型配置文件
│   └── ...
│
└── test_images/                   # 测试图像目录
```

---

## 🎨 界面说明

### GUI 界面截图

![GUI Screenshot](src/resources/GUI%20screenshot.png)

### 控制面板功能

- **File Loading**：加载图像序列
- **ROI Selection**：选择识别区域
- **Parameters**：时间、单位等参数设置
- **Pre-processing**：图像预处理选项
- **Processing**：开始/停止处理、导出数据
- **Interactive Chart**：置信度色彩编码的散点图，支持点击跳转到对应图像帧
- **Manual Correction**：手动校正识别结果（Ctrl+R 或双击图像）

---

## 📊 性能指标

| 指标             | 数值                 |
| ---------------- | -------------------- |
| **识别速度**     | 100-200 ms/图像      |
| **识别准确率**   | 97%+（SVTR_Tiny）    |
| **支持图像格式** | JPG, PNG, BMP, TIFF  |
| **最大图像序列** | 无限制（取决于内存） |
| **并发处理**     | 单线程异步处理       |

---

## 🛠️ 高级功能

### 置信度可视化与交互

**颜色编码系统**

程序自动为每个识别结果分配置信度评分，并通过颜色直观展示：

- **🟢 绿色点** (置信度 ≥ 0.9)

  - 表示高置信度识别
  - 这些结果高度可信，通常无需人工复核
  - 适合直接用于数据分析

- **🟠 橙色点** (置信度 0.75 ~ 0.89)

  - 表示中等置信度识别
  - 识别结果基本可信，但建议抽查验证
  - 如有异常值，需检查对应图像

- **🔴 红色点** (置信度 < 0.75)

  - 表示低置信度识别
  - 识别结果可信度较低，强烈建议人工复核
  - 可能是图像质量问题或数字模糊

- **▲ 黑色三角形** (NaN - 识别失败)
  - 表示完全无法识别
  - 使用三角形标记以区别于其他数据点
  - 需要检查原始图像
  - 可能需要调整预处理参数或 ROI 区域

**交互式图像跳转**

点击散点图上的任意数据点即可：

- 图像预览区域自动显示对应帧
- 图像滑块同步更新到对应位置
- 日志窗口记录跳转信息，包括：
  - 帧号 (例：15/100)
  - 时间值 (例：14.5 s)
  - 读数值 (例：25.3 °C)
  - 置信度 (例：0.876)

**应用场景**

1. **快速质检**：点击异常值（红色点/黑色三角形）快速定位问题图像
2. **数据验证**：点击橙色点进行抽样复核
3. **趋势分析**：观察颜色分布了解整体识别质量
4. **问题诊断**：集中的低置信度点可能表明该时段图像质量问题

**注意事项**

- 图像处理过程中点击跳转功能被禁用，防止线程冲突
- 处理完成后即可使用交互功能
- 导出的 Excel 文件包含完整的置信度列，便于后续分析

### 手动校正功能

当识别结果有误时，可使用手动校正功能进行修正：

**触发方式**

| 方式   | 操作                        |
| ------ | --------------------------- |
| 快捷键 | 导航到目标图像后按 `Ctrl+R` |
| 双击   | 双击预览窗口中的图像        |

**校正流程**

1. 导航到需要校正的图像帧
2. 使用上述任一方式打开校正对话框
3. 输入正确的数值（支持整数、小数、负数）
4. 确认后，程序自动更新：
   - 数据列表中的读数值
   - 散点图上对应的数据点位置
   - 预览窗口中的标注显示
   - 置信度设为 1.0（100%）

**使用场景**

- 修正个别识别错误的数值
- 处理特殊情况（如数字被部分遮挡）
- 快速补全识别失败的结果

**注意事项**

- 仅在批量处理完成后可用
- 处理过程中无法进行手动校正
- 校正后的值在导出 Excel 时会被采用

### 自定义预处理

根据实际情况调整预处理参数：

- **噪声较多**：增大 Erosion Size（如 2 或 3）进行腐蚀去噪
  - 启用后会自动处理小数点，替换为自适应圆点
- **断笔严重**：增大 Closing Size（如 3 或 5）进行闭运算连接
- **光照不均**：优化光照条件，或使用 Erosion Size 和 Closing Size 组合
- **参数说明**：
  - Erosion Size：0 表示禁用，>0 表示启用（使用 disk 形状核）
    - 启用后自动检测小数点并替换为圆点（直径=max(4× 腐蚀核, ROI 高度/15)）
  - Closing Size：0 表示禁用，>0 表示启用（使用矩形核，自动保护小数点）

### 小数点位置调整功能

当数码管显示不包含小数点，但实际读数需要小数位时，可使用此功能：

**应用场景**：

- 数码管显示 "12345"，实际读数为 "123.45"（两位小数）
- 数码管显示 "1000"，实际读数为 "1.000"（三位小数）
- 需要统一不同图像的小数位数格式

**使用方法**：

1. 在 **Decimal Position** 下拉框中选择目标精度
2. 程序会自动将识别结果调整为指定精度
3. 例如：识别结果 "12345" + 精度 "0.01" = 输出 "123.45"

**精度选项说明**：

- **Keep**：保持原始识别结果（默认，无调整）
- **None**：移除所有小数点，得到整数
- **0.1**：调整为 1 位小数（如 123.4）
- **0.01**：调整为 2 位小数（如 123.45）
- **0.001**：调整为 3 位小数（如 123.456）
- **0.0001**：调整为 4 位小数（如 123.4567）

**注意事项**：

- 此功能在识别后应用，不影响图像预处理
- 仅对数字格式的识别结果有效
- 调整会按比例缩放数值，保持相对精度

### 批量处理技巧

1. 按时间顺序命名图像文件
2. 确保所有图像中数码管位置一致
3. 使用更高分辨率的图像以提高识别率
4. 使用 Preview Preprocessing 验证预处理效果

---

## 🐛 故障排查

### 常见问题

**Q1: 识别率低怎么办？**

A: 尝试以下方法：

- 提高图像质量和分辨率
- 优化 ROI 选择（紧凑，边距适当）
- 调整预处理参数：
  - 噪声多：设置 Erosion Size > 0（如 2 或 3）
  - 断笔严重：设置 Closing Size > 0（如 3 或 5）
  - 两者可组合使用
- 改善拍摄条件（光照均匀、避免反光）

**Q2: 无法提取单个数字**

A: 可能原因：

- ROI 选择过大，包含多个数字
- 数字之间粘连：设置 Closing Size > 0（如 3 或 5）
- 背景噪声：设置 Erosion Size > 0（如 2 或 3）进行去噪
- 注意：启用 Erosion Size 后，小数点会被自动替换为圆点

**Q3: 小数点识别失败**

A:

- 确保小数点清晰可见且在 ROI 内
- 使用更高分辨率的图像
- 拍摄时更靠近数码管
- 启用 Erosion Size 后，程序会自动检测并替换小数点为圆点，提高识别率

**Q4: 程序启动失败**

A:

- 确认 Python 版本 >= 3.8
- 确认模型文件存在：检查 `./svtr_tiny_digital/` 目录中是否有 `best_accuracy.pdparams` 文件
- 确认字典文件存在：检查项目根目录是否有 `digital_dict.txt` 文件
- 重新安装依赖：`pip install -r requirements.txt --upgrade`
- Windows 用户：确认安装了 Visual C++ Redistributable

**Q5: 模型文件缺失**

A:

- 检查 `./svtr_tiny_digital/` 目录是否完整
- 确保包含以下文件：
  - **推理模型**（推荐，速度更快，部署优化）：
    - `inference.pdmodel`（模型结构）
    - `inference.pdiparams`（模型权重，~24MB）
  - **训练模型**（备用）：
    - `best_accuracy.pdparams`（模型权重，~24MB）
  - **配置文件**：
    - `config.yml`（模型配置）
- 程序会自动优先使用推理模型（inference.pdmodel + inference.pdiparams），如推理模型不存在则使用训练模型（best_accuracy.pdparams）
- 如果文件缺失，请从项目仓库重新下载或参考模型训练指南
- 同时确保项目根目录存在 `digital_dict.txt` 字典文件

### 日志分析

日志颜色含义：

- **黑色**：正常信息（INFO）
- **橙色**：警告信息（WARNING）
- **红色**：错误信息（ERROR）

---

## 🤝 贡献指南

欢迎贡献代码、报告 Bug 或提出新功能建议！

### 贡献流程

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

### 开发规范

- 遵循 PEP 8 代码规范
- 添加详细的注释和文档字符串
- 提交前运行测试

---

## 📄 许可证

本项目采用 **MIT License** 开源协议。详见 [LICENSE](LICENSE) 文件。

```
Copyright (c) 2025 Lucien

Version: 5.0.0
Last Updated: 2025-12-25

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 👨‍💻 作者信息

**Lucien**

- Email: lucien-6@qq.com
- GitHub: [@Lucien](https://github.com/yourusername)

---

## 🙏 致谢

- OpenCV 团队提供的强大计算机视觉库
- Python 社区的优秀开源工具
- 所有测试用户的反馈和建议

---

## 📮 联系方式

如有问题或建议，请通过以下方式联系：

- **Email**: lucien-6@qq.com
- **Issues**: [GitHub Issues](https://github.com/yourusername/Device_Number_Reading/issues)

---

<a name="english"></a>

## 📖 Introduction (English)

Device Number Reading is an advanced image recognition and analysis system designed for industrial device digital display readings. Using PaddleOCR SVTR_Tiny custom-trained model and computer vision technology, it automatically recognizes 7-segment/8-segment LED display numbers with high accuracy and exports time-series data to Excel format.

### ✨ Key Features

- 🎯 **High Precision**: Custom-trained SVTR_Tiny model for digital displays (97%+ accuracy)
- 🚀 **Ready to Use**: Pre-trained model included, no download required (24MB)
- 📦 **Lightweight**: 7x smaller than general models, faster inference
- 🔧 **Flexible Configuration**: Multiple preprocessing options and adjustable parameters
- 🔢 **Decimal Position Adjustment**: Adjust decimal position and precision after recognition
- 📊 **Smart Data Visualization**: Real-time scatter plot with confidence-based color coding
- 🎨 **Confidence Visualization**: Color-coded results (green=high, orange=medium, red=low, black=failed)
- 🖱️ **Interactive Chart**: Click data points to jump to corresponding image frames
- ✏️ **Manual Correction**: Edit recognition results manually (Ctrl+R or double-click)
- 💾 **Data Export**: One-click Excel export with confidence information
- 🎓 **Training Data Export**: Export results as PaddleOCR training data format (Ctrl+T)
- 🖼️ **ROI Selection**: Visual region of interest selection
- 🚀 **Batch Processing**: Support for image sequence batch recognition

### 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/Device_Number_Reading.git
cd Device_Number_Reading

# Install dependencies
pip install -r requirements.txt

# Note: The project uses pre-trained SVTR_Tiny digital-specific model (included)
# Model files are located in ./svtr_tiny_digital/ directory (~24MB)
# Inference model (recommended, faster):
#   - inference.pdmodel (model structure)
#   - inference.pdiparams (model weights)
# Training model (backup):
#   - best_accuracy.pdparams (model weights)
# The program automatically prioritizes inference model, falls back to training model if not available
# No additional download required, ready to use

# Run application
python Device_Reading_Analyzer.py
```

### 📘 Basic Workflow

1. Load image sequence
2. Select ROI (Region of Interest)
3. Configure parameters
4. Process images
5. Review & correct (Ctrl+R or double-click)
6. Export to Excel
7. Export training data (Ctrl+T, optional)

### 🔧 Technology Stack

- **Python 3.8+**: Main programming language
- **OpenCV 4.5+**: Image processing
- **PaddleOCR 2.9.1**: OCR inference engine
- **PaddlePaddle 2.6.2**: Deep learning framework
- **SVTR_Tiny**: Custom-trained digital recognition model
- **NumPy 1.24+**: Numerical computing
- **Tkinter**: GUI framework
- **Matplotlib 3.3+**: Data visualization
- **Pandas 1.3+**: Data processing

### 📊 Performance Metrics

- **Recognition Speed**: 100-200 ms/image
- **Accuracy**: 97%+ (SVTR_Tiny custom model)
- **Supported Formats**: JPG, PNG, BMP, TIFF
- **Max Image Sequence**: Unlimited (memory dependent)

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 👨‍💻 Author

**Lucien**

- Email: lucien-6@qq.com

---

<div align="center">

**⭐ If this project helps you, please give it a star! ⭐**

Made with ❤️ by Lucien

</div>
