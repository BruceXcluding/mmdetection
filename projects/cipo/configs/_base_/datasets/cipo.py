# CIPO数据集配置文件

# 数据集设置
dataset_type = 'CIPODataset'
data_root = '/Users/yigex/Documents/LLM-Inftra/mmdetection/data/cipo'  # 🔧 改为MMDetection内的数据目录
scene_file = '/Users/yigex/Documents/LLM-Inftra/OpenLane_CIPO/data/cipo/scene.json'

# 数据预处理管道
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')  # 🔧 简化预处理，避免复杂增强导致的问题
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor'))
]

test_pipeline = val_pipeline

# 数据加载器配置
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/training_coco.json',
        data_prefix=dict(img='images/training'),  # 🔧 修正：使用data_prefix而不是img_prefix
        scene_file=scene_file,
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/validation_coco.json',
        data_prefix=dict(img='images/validation'),  # 🔧 修正：使用data_prefix
        scene_file=scene_file,
        pipeline=val_pipeline))

test_dataloader = val_dataloader

# 评估器配置
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/annotations/validation_coco.json',
    metric='bbox',
    format_only=False)

test_evaluator = val_evaluator

# 类别数量
num_classes = 4