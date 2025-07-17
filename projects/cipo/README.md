# CIPO Detection with MMDetection

<div align="center">

[![MMDetection](https://img.shields.io/badge/MMDetection-3.x-orange)](https://github.com/open-mmlab/mmdetection)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

**基于MMDetection的CIPO数据集目标检测训练框架**

[快速开始](#快速开始) • [模型训练](#模型训练) • [评估测试](#评估测试) • [常见问题](#常见问题)

</div>

---

## 📋 目录

- [项目简介](#项目简介)
- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
- [模型训练](#模型训练)
- [评估测试](#评估测试)
- [模型库](#模型库)
- [配置说明](#配置说明)
- [常见问题](#常见问题)
- [更新日志](#更新日志)

---

## 🎯 项目简介

本项目基于[MMDetection](https://github.com/open-mmlab/mmdetection)框架，专为OpenLane **CIPO**（Collision-Induced Pose Occlusion）数据集开发的目标检测解决方案。

### 🔬 CIPO数据集特点
- **4个困难等级**: Level 1-4，从简单到复杂的遮挡场景
- **4种目标类型**: `vehicle`, `pedestrian`, `sign`, `cyclist`
- **多场景覆盖**: residential, urban, highway, parking等
- **复杂环境**: 不同天气、时间、光照条件

### ✨ 项目特色
- 🚀 **开箱即用**: 基于MMDetection成熟框架
- 🎯 **专门优化**: 针对CIPO数据集特点定制
- 📊 **多模型支持**: RTMDet, YOLO, Faster R-CNN等
- 🔧 **灵活配置**: 易于调整和扩展
- 📈 **完整流程**: 数据处理→训练→评估→部署

---

## 🛠 环境配置

### 系统要求
- **操作系统**: Linux, macOS, Windows
- **Python**: 3.8+
- **PyTorch**: 1.8+
- **CUDA**: 11.0+ (GPU训练)

### 安装步骤

#### 1. 安装MMDetection
```bash
# 进入项目目录
cd /Users/yigex/Documents/LLM-Inftra/mmdetection

# 安装依赖
pip install -r requirements.txt
pip install mmengine mmcv==2.0.0rc4

# 安装MMDetection (开发模式)
pip install -v -e .

# 验证安装
python tools/misc/print_config.py configs/rtmdet/rtmdet_l_8xb32-300e_coco.py
```

#### 2. 安装额外依赖
```bash
# CIPO项目特定依赖
pip install tqdm pillow pathlib2
```

#### 3. 验证环境
```bash
# 测试MMDetection
python -c "import mmdet; print(f'MMDetection: {mmdet.__version__}')"

# 测试GPU (可选)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 📁 项目结构

```
mmdetection/
├── projects/
│   └── cipo/                                    # 🎯 CIPO项目根目录
│       ├── README.md                            # 本文档
│       ├── configs/                             # 配置文件目录
│       │   ├── _base_/                          # 基础配置
│       │   │   └── datasets/
│       │   │       └── cipo.py                  # 📋 CIPO数据集配置
│       │   ├── rtmdet_s_8xb32-300e_cipo.py     # RTMDet-S配置
│       │   ├── rtmdet_l_8xb32-300e_cipo.py     # 🚀 RTMDet-L配置
│       │   ├── yolov8_l_8xb16-500e_cipo.py     # YOLOv8配置
│       │   └── faster_rcnn_r50_fpn_1x_cipo.py  # Faster R-CNN配置
│       ├── cipo_dataset.py                      # 数据集实现
│       ├── convert_cipo_standalone.py           # 独立数据转换脚本
│       ├── train_cipo.py                        # 训练脚本
│       └── eval_cipo.py                         # 评估脚本
├── data/                                        # 数据目录
│   └── cipo/                                    # CIPO数据
│       ├── annotations/                         # COCO格式标注
│       │   ├── training_coco.json               # 训练集标注
│       │   └── validation_coco.json             # 验证集标注
│       ├── images/                              # 图像目录(软链接)
│       │   ├── training/
│       │   └── validation/
│       └── cipo.yaml                            # 数据集配置
└── work_dirs/                                   # 训练输出目录
    ├── cipo_rtmdet_l/                           # RTMDet-L训练结果
    ├── cipo_yolov8_l/                           # YOLOv8训练结果
    └── ...
```

---

## 📊 数据准备

### 1. 原始数据结构
确保您的CIPO数据集结构如下：
```
/path/to/original/cipo/
├── training/
│   ├── segment-xxx/
│   │   ├── frame1.json
│   │   ├── frame2.json
│   │   └── ...
│   └── segment-yyy/
├── validation/
│   └── segment-zzz/
├── images/
│   ├── training/
│   │   ├── segment-xxx/
│   │   │   ├── frame1.jpg
│   │   │   └── frame2.jpg
│   │   └── segment-yyy/
│   └── validation/
└── scene.json              # 场景标签文件(可选)
```

### 2. 数据转换为COCO格式

#### 方法1: 使用独立转换脚本 (推荐)
```bash
cd projects/cipo

# 转换数据
python convert_cipo_standalone.py
```

#### 方法2: 使用数据集类转换
```bash
cd projects/cipo

# 如果MMDetection环境正常
python cipo_dataset.py
```

### 3. 验证数据转换
```bash
# 检查生成的文件
ls -la data/cipo/annotations/
# training_coco.json  validation_coco.json

# 检查图像链接
ls -la data/cipo/images/
# training/  validation/

# 验证JSON格式
python -c "
import json
with open('data/cipo/annotations/training_coco.json', 'r') as f:
    data = json.load(f)
print(f'Images: {len(data[\"images\"])}, Annotations: {len(data[\"annotations\"])}')
"
```

---

## 🚀 快速开始

### 1. 训练基线模型 (RTMDet-S)
```bash
# 快速验证流程
python tools/train.py projects/cipo/configs/rtmdet_s_8xb32-300e_cipo.py \
    --work-dir ./work_dirs/quick_test
```

### 2. 训练完整模型 (RTMDet-L)
```bash
# 完整训练
python tools/train.py projects/cipo/configs/rtmdet_l_8xb32-300e_cipo.py \
    --work-dir ./work_dirs/cipo_rtmdet_l
```

### 3. 验证训练结果
```bash
# 在验证集上测试
python tools/test.py \
    projects/cipo/configs/rtmdet_l_8xb32-300e_cipo.py \
    work_dirs/cipo_rtmdet_l/latest.pth
```

---

## 🎯 模型训练

### 单GPU训练
```bash
# RTMDet-L (推荐)
python tools/train.py projects/cipo/configs/rtmdet_l_8xb32-300e_cipo.py \
    --work-dir ./work_dirs/cipo_rtmdet_l

# YOLOv8-L
python tools/train.py projects/cipo/configs/yolov8_l_8xb16-500e_cipo.py \
    --work-dir ./work_dirs/cipo_yolov8_l

# Faster R-CNN
python tools/train.py projects/cipo/configs/faster_rcnn_r50_fpn_1x_cipo.py \
    --work-dir ./work_dirs/cipo_faster_rcnn
```

### 多GPU训练
```bash
# 使用4张GPU训练
bash tools/dist_train.sh \
    projects/cipo/configs/rtmdet_l_8xb32-300e_cipo.py \
    4 \
    --work-dir ./work_dirs/cipo_rtmdet_l_4gpu

# 使用8张GPU训练  
bash tools/dist_train.sh \
    projects/cipo/configs/rtmdet_l_8xb32-300e_cipo.py \
    8 \
    --work-dir ./work_dirs/cipo_rtmdet_l_8gpu
```

### 训练参数调整
```bash
# 启用混合精度训练 (节省显存)
python tools/train.py projects/cipo/configs/rtmdet_l_8xb32-300e_cipo.py \
    --amp \
    --work-dir ./work_dirs/cipo_rtmdet_l_amp

# 自动缩放学习率
python tools/train.py projects/cipo/configs/rtmdet_l_8xb32-300e_cipo.py \
    --auto-scale-lr \
    --work-dir ./work_dirs/cipo_rtmdet_l_auto_lr

# 从checkpoint继续训练
python tools/train.py projects/cipo/configs/rtmdet_l_8xb32-300e_cipo.py \
    --resume work_dirs/cipo_rtmdet_l/latest.pth
```

### 训练监控
```bash
# 实时查看训练日志
tail -f work_dirs/cipo_rtmdet_l/*.log

# 使用TensorBoard监控 (需要安装tensorboard)
pip install tensorboard
tensorboard --logdir work_dirs/cipo_rtmdet_l/tf_logs
```

---

## 📈 评估测试

### 标准评估
```bash
# 在验证集上评估
python tools/test.py \
    projects/cipo/configs/rtmdet_l_8xb32-300e_cipo.py \
    work_dirs/cipo_rtmdet_l/best_coco_bbox_mAP_epoch_*.pth

# 评估多个checkpoint
python tools/test.py \
    projects/cipo/configs/rtmdet_l_8xb32-300e_cipo.py \
    work_dirs/cipo_rtmdet_l/epoch_*.pth \
    --work-dir ./eval_results
```

### 生成预测结果
```bash
# 保存预测结果为JSON
python tools/test.py \
    projects/cipo/configs/rtmdet_l_8xb32-300e_cipo.py \
    work_dirs/cipo_rtmdet_l/latest.pth \
    --format-only \
    --eval-options "jsonfile_prefix=./results/cipo_predictions"
```

### CIPO官方评估
```bash
# 生成CIPO评估格式
python projects/cipo/eval_cipo.py \
    --config projects/cipo/configs/rtmdet_l_8xb32-300e_cipo.py \
    --checkpoint work_dirs/cipo_rtmdet_l/latest.pth \
    --output-dir ./cipo_eval_results

# 使用CIPO官方评估脚本
python /path/to/OpenLane/eval/CIPO_evaluation/example/EvalDemo.py \
    --anno_txt ./cipo_eval_results/annotation_files.txt \
    --res_txt ./cipo_eval_results/result_files.txt
```

### 可视化结果
```bash
# 单张图像推理
python demo/image_demo.py \
    /path/to/test/image.jpg \
    projects/cipo/configs/rtmdet_l_8xb32-300e_cipo.py \
    work_dirs/cipo_rtmdet_l/latest.pth \
    --out-dir ./demo_results

# 批量图像推理
python demo/image_demo.py \
    /path/to/test/images/ \
    projects/cipo/configs/rtmdet_l_8xb32-300e_cipo.py \
    work_dirs/cipo_rtmdet_l/latest.pth \
    --out-dir ./demo_results
```

---

## 🏆 模型库

### 基准模型性能

| 模型 | 配置 | mAP | mAP@0.5 | mAP@0.75 | 下载 |
|------|------|-----|---------|----------|------|
| RTMDet-S | [config](configs/rtmdet_s_8xb32-300e_cipo.py) | 待测试 | 待测试 | 待测试 | [model](https://github.com/your-repo/releases) |
| RTMDet-L | [config](configs/rtmdet_l_8xb32-300e_cipo.py) | 待测试 | 待测试 | 待测试 | [model](https://github.com/your-repo/releases) |
| YOLOv8-L | [config](configs/yolov8_l_8xb16-500e_cipo.py) | 待测试 | 待测试 | 待测试 | [model](https://github.com/your-repo/releases) |
| Faster R-CNN | [config](configs/faster_rcnn_r50_fpn_1x_cipo.py) | 待测试 | 待测试 | 待测试 | [model](https://github.com/your-repo/releases) |

### 不同CIPO Level性能分析

| Level | 难度 | 特点 | RTMDet-L mAP | 备注 |
|-------|------|------|--------------|------|
| Level 1 | 简单 | 无遮挡或轻微遮挡 | 待测试 | 基准性能 |
| Level 2 | 中等 | 部分遮挡 | 待测试 | 常见场景 |
| Level 3 | 困难 | 严重遮挡 | 待测试 | 挑战场景 |
| Level 4 | 极难 | 极度遮挡 | 待测试 | 边缘情况 |

---

## ⚙️ 配置说明

### 数据集配置 (`configs/_base_/datasets/cipo.py`)
```python
# 关键配置项
dataset_type = 'CIPODataset'
data_root = '/path/to/mmdetection/data/cipo'
num_classes = 4  # vehicle, pedestrian, sign, cyclist

# 数据增强配置
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
```

### 模型配置示例
```python
# RTMDet-L配置
model = dict(
    type='RTMDet',
    bbox_head=dict(
        num_classes=4,  # CIPO类别数
        loss_cls=dict(type='QualityFocalLoss'),
        loss_bbox=dict(type='GIoULoss')
    )
)

# 训练配置
max_epochs = 300
train_cfg = dict(max_epochs=max_epochs, val_interval=10)
```

### 自定义配置
```bash
# 创建自定义配置
cp projects/cipo/configs/rtmdet_l_8xb32-300e_cipo.py \
   projects/cipo/configs/rtmdet_l_custom.py

# 修改关键参数
# - 学习率: optimizer.lr
# - 批量大小: train_dataloader.batch_size  
# - 训练轮数: max_epochs
# - 数据增强: train_pipeline
```

---

## 🔧 常见问题

### 环境问题

**Q: Mac M系列芯片MMDetection安装失败？**
```bash
# A: 使用独立转换脚本
cd projects/cipo
python convert_cipo_standalone.py

# 然后使用YOLOv8等Mac友好的框架
pip install ultralytics
```

**Q: CUDA版本不匹配？**
```bash
# A: 检查并重新安装对应版本
python -c "import torch; print(torch.version.cuda)"
pip install torch torchvision -f https://download.pytorch.org/whl/cu116/torch_stable.html
```

### 数据问题

**Q: 数据转换失败？**
```bash
# A: 检查原始数据结构
ls /path/to/original/cipo/
# 确保包含 training/, validation/, images/ 目录

# 检查场景文件
cat /path/to/original/cipo/scene.json
```

**Q: 图像加载错误？**
```bash
# A: 验证软链接
ls -la data/cipo/images/
# 如果软链接失败，手动复制
cp -r /path/to/original/cipo/images/* data/cipo/images/
```

### 训练问题

**Q: 显存不足？**
```bash
# A: 减少批量大小
# 修改配置文件中的 train_dataloader.batch_size = 2

# 启用混合精度训练
python tools/train.py config.py --amp
```

**Q: 训练速度慢？**
```bash
# A: 使用多GPU训练
bash tools/dist_train.sh config.py 4

# 减少验证频率
# 修改配置中的 val_interval = 20
```

**Q: 模型不收敛？**
```bash
# A: 检查学习率设置
# 减小学习率: optimizer.lr = 0.001

# 检查数据质量
python tools/misc/browse_dataset.py config.py
```

### 评估问题

**Q: mAP过低？**
- 检查数据标注质量
- 调整数据增强策略
- 尝试不同的模型架构
- 增加训练轮数

**Q: 官方评估格式转换失败？**
```bash
# A: 使用官方评估工具
cd /path/to/OpenLane/eval/CIPO_evaluation/example/
python EvalDemo.py --anno_txt anno.txt --res_txt result.txt
```

---

## 📝 开发指南

### 添加新模型
1. 在 `configs/` 目录创建新配置文件
2. 继承基础数据集配置: `_base_ = ['./_base_/datasets/cipo.py']`
3. 定义模型架构和训练参数
4. 测试配置: `python tools/misc/print_config.py new_config.py`

### 自定义数据增强
```python
# 在配置文件中修改 train_pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),  # 新增
    dict(type='RandomCrop', crop_size=(0.8, 0.8)),  # 新增
    dict(type='PackDetInputs')
]
```

### 添加新评估指标
```python
# 在配置文件中修改 val_evaluator
val_evaluator = [
    dict(type='CocoMetric', metric='bbox'),
    dict(type='CIPOMetric', metric='cipo_levels')  # 自定义指标
]
```

---

## 📜 许可证

本项目采用 [Apache License 2.0](https://github.com/open-mmlab/mmdetection/blob/master/LICENSE) 许可证。

---

## 🙏 致谢

- [MMDetection](https://github.com/open-mmlab/mmdetection) 提供优秀的检测框架
- [OpenLane](https://github.com/OpenDriveLab/OpenLane) 提供CIPO数据集
- 所有贡献者和测试用户

---

<div align="center">

**🌟 如果这个项目对您有帮助，请给我们一个Star！**

[⬆️ 回到顶部](#cipo-detection-with-mmdetection)

</div>
