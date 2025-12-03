# 快速开始指南 | Quick Start Guide

## 🚀 5分钟上手教程

### 第一步：安装程序 (1分钟)

```bash
# 安装依赖
pip install -r requirements.txt

# 注意：本项目使用 SVTR_Tiny 数码管专用模型（已包含在项目中）
# 推理模型位于 ./svtr_tiny_digital/ 目录（约 24MB）
# 包含推理模型（inference.pdmodel + inference.pdiparams）和训练模型（best_accuracy.pdparams）
# 程序自动优先使用推理模型，速度更快、部署优化
# 无需额外下载，开箱即用

# 启动程序
python Device_Reading_Analyzer.py
```

### 第二步：准备图像 (1分钟)

准备一个包含设备数码显示读数的图像序列文件夹，例如：

```
my_readings/
├── image_001.jpg
├── image_002.jpg
├── image_003.jpg
└── ...
```

**要求**：
- 图像中数码管位置保持一致
- 图像清晰，数字可见
- 按时间顺序命名（可选）

### 第三步：加载并选择ROI (1分钟)

1. 点击 `Load Image Sequence` 按钮，选择图像文件夹
2. 点击 `Select ROI` 按钮
3. 在图像上拖动鼠标框选数码显示区域
4. 确保完整包含所有数字

![ROI选择示例](docs/roi_selection.png)

### 第四步：处理并导出 (30秒)

1. 设置参数：
   - Time Interval: `1.0`
   - Time Unit: `seconds`
   - Decimal Position: `Keep`（默认，或根据需要选择精度）

2. 点击 `Process Images` 开始处理

3. 如需修正识别结果：按 `Ctrl+R` 或双击图像进行手动校正

4. 处理完成后，点击 `Export to Excel`

**完成！** 🎉 您的数据已导出到Excel文件。

**Decimal Position 参数说明**：
- **Keep**：保持原始识别结果（推荐用于有小数点的显示）
- **None**：移除小数点，得到整数
- **0.1 / 0.01 / 0.001 / 0.0001**：调整为指定小数精度
- **使用场景**：当数码管不显示小数点，但实际读数需要小数位时使用

---

## 🎯 最佳实践

### 拍摄技巧

✅ **推荐做法**：
- 固定相机位置和角度
- 保持稳定光照条件
- 确保数字清晰对焦
- 避免反光和阴影

❌ **避免事项**：
- 相机抖动或角度变化
- 过度曝光或欠曝光
- 数字模糊不清
- 部分遮挡

### 参数调优

| 场景 | 推荐设置 |
|------|----------|
| **标准识别** | Erosion Size: 0, Closing Size: 0, Decimal Position: Keep，ROI紧凑 |
| **低质量图像** | Erosion Size: 2, Closing Size: 5, Decimal Position: Keep |
| **噪声较多** | Erosion Size: 2-3, Closing Size: 0, Decimal Position: Keep（自动处理小数点） |
| **断笔严重** | Erosion Size: 0, Closing Size: 3-5, Decimal Position: Keep |
| **光照不均** | Erosion Size: 2, Closing Size: 3, Decimal Position: Keep，优化光照条件 |
| **高精度要求** | 高分辨率图像，ROI高度>50像素，SVTR_Tiny推理模型，Decimal Position: Keep |
| **小数点识别困难** | Erosion Size: 2-3（自动替换为圆点），Decimal Position: Keep |
| **无小数点显示** | Erosion Size: 0, Closing Size: 0, Decimal Position: 0.01（或其他精度） |

---

## 🔍 示例场景

### 场景1：实验室数据采集

**需求**：记录温度计读数，每5秒一次，持续1小时

**设置**：
```
Time Interval: 5
Time Unit: seconds
Start Time: 0
Reading Unit: °C
Decimal Position: Keep
```

**结果**：Excel文件包含 720 行数据（1小时 = 720个5秒间隔）

**Decimal Position 应用示例**：
- 如果温度计显示 "2536"，但实际温度为 "25.36°C"
- 设置 Decimal Position: 0.01（两位小数）
- 程序自动将 "2536" 转换为 "25.36"

### 场景2：工业设备监控

**需求**：从历史监控录像中提取压力表读数

**步骤**：
1. 从视频中提取关键帧图像
2. 使用本工具批量识别
3. 分析压力变化曲线

### 场景3：历史数据恢复

**需求**：从老旧照片中提取数据

**建议**：
- 先用图像编辑软件增强对比度
- 设置 Erosion Size > 0 和 Closing Size > 0 进行预处理
- 启用 Erosion Size 后，小数点会自动替换为圆点，提高识别率
- 优化光照条件
- 使用更高分辨率的图像

---

## ❓ 常见问题速查

### Q: 识别结果不准确？

**A**: 尝试以下步骤：
1. 提高图像质量和分辨率
2. 优化ROI选择（紧凑，留适当边距）
3. 使用 `Preview Preprocessing` 检查预处理效果
   4. 调整预处理参数：
      - 噪声多：设置 Erosion Size > 0（如 2 或 3），会自动处理小数点
      - 断笔严重：设置 Closing Size > 0（如 3 或 5）

### Q: 部分图像无法识别？

**A**: 检查：
- [ ] 图像质量是否足够清晰
- [ ] ROI是否完整包含所有数字
- [ ] 数码管位置是否与训练样本一致
- [ ] 是否有反光或遮挡

### Q: 小数点识别失败？

**A**: 
- 确保小数点清晰可见且在ROI范围内
- 使用更高分辨率的图像
- 拍摄时更靠近数码管
- 启用 Erosion Size: 2-3，自动替换小数点为圆点
- SVTR_Tiny推理模型对小数点识别更准确

### Q: 小数点位置不对？

**A**：使用 Decimal Position 功能调整
- **问题场景**：数码管显示 "12345"，但实际读数应为 "123.45"
- **解决方案**：
  1. 在 Decimal Position 下拉框选择 "0.01"（两位小数）
  2. 程序自动将识别结果调整为正确精度
  3. 输出结果："123.45"
- **其他用途**：
  - 统一不同图像的小数位数格式
  - 去除小数点（选择 "None"）
  - 转换整数为小数（选择目标精度）

### Q: 个别识别结果有误？

**A**：使用手动校正功能
- **触发方式**：
  - 方式一：导航到目标图像后按 `Ctrl+R`
  - 方式二：双击预览窗口中的图像
- **校正步骤**：
  1. 在弹出对话框中输入正确数值
  2. 点击 `Confirm` 或按 `Enter` 确认
  3. 程序自动更新散点图和预览标注
- **注意事项**：
  - 仅在批量处理完成后可用
  - 校正后置信度自动设为 1.0 (100%)

---

## 📞 获取帮助

遇到问题？
1. 查看完整文档：[README.md](README.md)
2. 查看故障排查章节
3. 提交 Issue：[GitHub Issues](https://github.com/yourusername/Device_Number_Reading/issues)
4. 联系作者：lucien-6@qq.com

---

## 🎓 进阶学习

完成快速开始后，您可以：

- 📖 阅读完整的 [README.md](README.md) 了解技术细节
- 🔧 探索高级功能和参数调优
- 🤝 为项目贡献代码或建议
- 📊 结合其他工具进行数据分析

---

<div align="center">

**Happy Coding! 🚀**

如果本项目对您有帮助，请给个 ⭐ Star！

</div>

