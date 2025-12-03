# Device Readings Analyzer - 项目结构 (新版)

## 📁 项目目录结构

```
Device_Number_Reading/
│
├── 📄 Device_Reading_Analyzer.py          # 程序主入口（模块化架构，v5.0.0）
│
├── 📂 src/                                 # 源代码模块目录
│   ├── 📄 __init__.py                      # 包初始化
│   ├── 📄 main_window.py                   # 主窗口模块
│   │
│   ├── 📂 ui/                              # UI组件模块（预留）
│   │   └── 📄 __init__.py                  # 当前UI功能集成在main_window.py
│   │
│   ├── 📂 core/                            # 核心功能模块
│   │   ├── 📄 __init__.py
│   │   ├── 📄 image_processor.py           # 图像处理 ✨ 完整实现
│   │   └── 📄 digit_recognizer.py          # 数字识别（PaddleOCR）✨ 完整实现
│   │
│   ├── 📂 utils/                           # 工具模块（预留）
│   │   └── 📄 __init__.py
│   │
│   └── 📂 resources/                       # 资源文件 ✨ 完整实现
│       ├── 📄 __init__.py                  # 包初始化
│       ├── 📄 help_content_cn.txt          # 中文使用指南
│       └── 📄 help_content_en.txt          # 英文使用指南
│
├── 📂 test_images/                         # 测试图像目录
│
├── 📄 requirements.txt                     # 依赖包列表
├── 📄 README.md                            # 项目说明
├── 📄 QUICKSTART.md                        # 快速开始指南
├── 📄 ARCHITECTURE.md                      # 架构说明
├── 📄 REFACTORING_SUMMARY.md               # 重构总结
├── 📄 CHANGELOG.md                         # 更新日志
├── 📄 LICENSE                              # MIT许可证
└── 📄 .gitignore                           # Git忽略文件

```

## 🎯 使用方式

### 启动程序
```bash
python Device_Reading_Analyzer.py
```
- ✅ 模块化架构
- ✅ 所有功能完整
- ✅ 易于维护扩展
- ✅ 代码组织清晰

## 📊 模块统计

| 类别 | 模块数 | 状态 | 总行数 |
|------|--------|------|--------|
| 主窗口 | 1 | 完整实现 | ~1700 |
| 核心功能 | 2 | 2完整 | ~805 |
| 资源文件 | 2 | 2完整 | ~1100 |
| **总计** | **5** | **5完整** | **~3605** |

**说明**：
- UI功能已集成在 `main_window.py` 中，保持代码简洁
- 帮助文档已从代码中分离为独立资源文件（中英文各约550行）
- main_window.py 包含完整功能实现（约1700行）
- 所有功能完整可用，代码结构清晰，易于维护
- 版本：v5.0.0（2025-12-03）

## 🌟 完整实现的模块

1. **main_window.py** (~1700行)
   - 主窗口完整实现
   - UI界面和交互逻辑
   - 帮助文档加载（从独立资源文件读取中英文双语内容）
   - ROI选择功能
   - 图像处理和识别流程
   - **小数点位置调整功能**（Keep/None/0.1/0.01/0.001/0.0001）
   - **手动校正功能**（Ctrl+R 或双击图像触发）
   - 数据可视化和导出
   - 日志管理（集成）
   - 配置管理（集成）

2. **digit_recognizer.py** (~535行)
   - **SVTR_Tiny 数码管专用识别引擎**
   - **推理模型优先加载**（inference.pdmodel + inference.pdiparams）
   - 训练模型备用支持（best_accuracy.pdparams）
   - 自定义字典支持（0-9, -, .）
   - 图像预处理优化（组件间距扩展、尺寸调整）
   - 字符白名单过滤
   - 置信度阈值过滤
   - 数字格式验证
   - 详细日志回调

3. **image_processor.py** (~570行)
   - 图像预处理流程
   - 灰度化和二值化（Otsu自适应）
   - 小对象过滤
   - 形态学处理（腐蚀、闭运算）
   - **小数点自动处理机制**：
     - 检测小数点（面积最小且<倒数第二小的1/4）
     - 扩展间距（4倍腐蚀核大小）
     - 替换为实心圆点（自适应直径）
   - 组件过滤和边界添加

4. **help_content_cn.txt** (~550行)
   - 中文使用指南完整内容
   - 包含程序简介、使用流程、手动校正、常见问题等
   - 独立维护，便于更新

5. **help_content_en.txt** (~550行)
   - 英文使用指南完整内容
   - 包含程序简介、使用流程、手动校正、常见问题等
   - 独立维护，便于更新

## 🔄 重构策略

采用**渐进式重构**：

```
Phase 1: 建立架构 ✅
  └─ 创建目录结构
  └─ 创建__init__.py文件
  
Phase 2: 提取资源 ✅
  └─ 提取帮助文档
  └─ 分离配置信息
  
Phase 3: 创建工具 ✅
  └─ 日志管理系统
  └─ 配置管理系统
  
Phase 4-7: 模块化 ✅
  └─ 创建模块框架
  └─ 保持向后兼容
  
Phase 8: 测试优化 ✅
  └─ 验证模块导入
  └─ 功能测试
```

## 📝 模块说明

### 主窗口模块 (src/main_window.py)

**完整功能实现** ✨
- 主窗口UI界面（Tkinter）
- 图像加载和显示
- ROI选择功能（鼠标交互）
- 参数配置面板
- 图像预处理预览
- 批量图像处理
- 实时识别结果显示
- 数据可视化（Matplotlib）
- **手动校正功能**（Ctrl+R 或双击图像）
- Excel数据导出
- **导出训练数据**（Ctrl+T，PaddleOCR格式）
- 帮助窗口（中英文双语，动态加载资源文件）
- 日志显示和管理

### 核心模块 (src/core/)

**digit_recognizer.py** ✨
- **SVTR_Tiny 数码管专用识别引擎**（v5.0.0）
- **推理模型优先支持**：
  - inference.pdmodel + inference.pdiparams（推荐，速度更快）
  - best_accuracy.pdparams（训练模型，备用）
  - 自动检测并选择最优模型
- 自定义字典加载（digital_dict.txt）
- 懒加载模型初始化
- 图像预处理优化（组件间距扩展、尺寸调整）
- 字符白名单过滤（"-0123456789."）
- 置信度阈值过滤
- 数字格式验证
- OCR结果可视化保存

**image_processor.py** ✨
- 灰度化转换
- Otsu自适应二值化
- 小对象过滤
- **小数点自动处理**（v3.0.2+）：
  - 自动检测小数点组件
  - 扩展水平间距（4倍腐蚀核）
  - 替换为实心圆点（自适应大小）
- 形态学处理（腐蚀、闭运算）
- 边界添加优化

### 工具模块 (src/utils/)

**预留模块** - 用于未来扩展的工具函数和类

### 资源模块 (src/resources/)

**help_content_cn.txt** ✨
- 中文使用指南完整内容（~550行）
- UTF-8编码格式
- 包含程序简介、使用流程、手动校正、参数说明、常见问题等
- 独立维护，便于更新和国际化

**help_content_en.txt** ✨
- 英文使用指南完整内容（~550行）
- UTF-8编码格式
- 包含程序简介、使用流程、手动校正、参数说明、常见问题等
- 独立维护，便于更新和国际化

### 预留模块

**src/ui/** - UI组件模块（预留用于未来重构）

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行程序
```bash
# 使用原始版本
python Device_Reading_Analyzer.py

# 或使用模块化版本
python Device_Reading_Analyzer_modular.py
```

### 3. 查看帮助
程序运行后，点击菜单栏：`Help → User Guide`

## 📖 文档

- **README.md** - 项目介绍和使用说明
- **QUICKSTART.md** - 快速开始指南
- **ARCHITECTURE.md** - 架构设计说明 ✨ 新增
- **REFACTORING_SUMMARY.md** - 重构总结 ✨ 新增
- **CHANGELOG.md** - 版本更新日志

## ✅ 验证状态

所有核心模块功能完整可用 ✅

```
✅ Main entry point (Device_Reading_Analyzer.py) ... OK
✅ Main window module (main_window.py) ............. OK
✅ Image processor module .......................... OK
✅ Digit recognizer module ......................... OK
✅ Config module ................................... OK
✅ Logger module ................................... OK
```

**代码质量**：
- ✅ 无冗余代码
- ✅ 注释准确完整
- ✅ 版本信息一致
- ✅ 无linter错误

## 🔮 未来计划

### 短期
- [x] 代码清理和优化（已完成）
- [x] 移除冗余模块（已完成）
- [ ] 添加单元测试
- [ ] 性能优化

### 中期
- [ ] 增加类型注解
- [ ] API文档
- [ ] 可选的UI模块化重构

### 长期
- [ ] 插件系统
- [ ] 配置文件持久化
- [ ] 更多语言支持

## 👤 作者

- **开发者**: Lucien
- **邮箱**: lucien-6@qq.com
- **许可证**: MIT License
- **版本**: 5.0.0
- **更新日期**: 2025-12-03

---

**状态**: ✅ 所有功能完整可用，代码已清理优化，帮助文档已分离为独立资源文件

