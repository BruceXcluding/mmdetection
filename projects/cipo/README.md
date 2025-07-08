mmdetection/
├── projects/
│   └── cipo/                           # 🎯 CIPO项目目录
│       ├── configs/                    # 配置文件目录
│       │   ├── _base_/                 # 基础配置
│       │   │   └── datasets/
│       │   │       └── cipo.py         # 📋 CIPO数据集配置
│       │   └── rtmdet_l_8xb32-300e_cipo.py  # 🚀 RTMDet-L完整配置
│       ├── cipo_dataset.py             # 数据集类实现
│       └── train_cipo.py               # 训练脚本
├── data/
│   └── cipo/                           # 数据目录
│       ├── annotations/                # COCO格式标注
│       │   ├── training_coco.json
│       │   └── validation_coco.json
│       └── images/                     # 图像软链接
│           ├── training/
│           └── validation/
└── work_dirs/                          # 训练输出目录
    └── cipo_rtmdet_l/
        ├── rtmdet_l_8xb32-300e_cipo.py # 训练时的完整配置
        ├── latest.pth                   # 最新checkpoint
        ├── best_coco_bbox_mAP_epoch_X.pth  # 最佳模型
        └── 20231208_120000.log         # 训练日志


cd /Users/yigex/Documents/LLM-Inftra/mmdetection

# 安装依赖
pip install -r requirements.txt
pip install -v -e .

# 验证安装
python tools/misc/print_config.py configs/rtmdet/rtmdet_l_8xb32-300e_coco.py

# 创建项目目录
mkdir -p projects/cipo/configs/_base_/datasets
mkdir -p data/cipo

# 转换数据格式
cd projects/cipo
python cipo_dataset.py

# 快速开始 - RTMDet-L
python projects/cipo/train_cipo.py projects/cipo/configs/rtmdet_l_8xb32-300e_cipo.py

# 自定义工作目录
python projects/cipo/train_cipo.py \
    projects/cipo/configs/rtmdet_l_8xb32-300e_cipo.py \
    --work-dir ./work_dirs/cipo_rtmdet_l

# 启用混合精度训练
python projects/cipo/train_cipo.py \
    projects/cipo/configs/rtmdet_l_8xb32-300e_cipo.py \
    --amp

# 多GPU训练
bash tools/dist_train.sh \
    projects/cipo/configs/rtmdet_l_8xb32-300e_cipo.py \
    8 --work-dir ./work_dirs/cipo_rtmdet_l

# 在验证集上评估
python tools/test.py \
    projects/cipo/configs/rtmdet_l_8xb32-300e_cipo.py \
    work_dirs/cipo_rtmdet_l/latest.pth

# 生成CIPO评估格式的结果
python tools/test.py \
    projects/cipo/configs/rtmdet_l_8xb32-300e_cipo.py \
    work_dirs/cipo_rtmdet_l/latest.pth \
    --format-only \
    --eval-options "jsonfile_prefix=./results/cipo_results"

# 图像推理演示
python demo/image_demo.py \
    /path/to/test/image.jpg \
    projects/cipo/configs/rtmdet_l_8xb32-300e_cipo.py \
    work_dirs/cipo_rtmdet_l/latest.pth \
    --out-dir ./demo_results