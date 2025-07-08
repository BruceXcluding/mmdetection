# CIPO数据集适配器 - 用于MMDetection
import os
import json
import numpy as np
from pathlib import Path
from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS
import mmcv
from pycocotools.coco import COCO
import tempfile


@DATASETS.register_module()
class CIPODataset(CocoDataset):
    """CIPO数据集，继承自CocoDataset"""
    
    # CIPO类别定义
    METAINFO = {
        'classes': ('vehicle', 'pedestrian', 'sign', 'cyclist'),
        'palette': [
            (220, 20, 60),    # vehicle - 红色
            (119, 11, 32),    # pedestrian - 深红色  
            (0, 0, 142),      # sign - 蓝色
            (0, 0, 230),      # cyclist - 深蓝色
        ]
    }
    
    def __init__(self, 
                 ann_file='',
                 data_prefix=dict(img=''),  # 🔧 修正：使用data_prefix
                 scene_file=None,
                 cipo_level_weight=1.0,
                 **kwargs):
        
        self.scene_file = scene_file
        self.cipo_level_weight = cipo_level_weight
        self.scene_tags = {}
        
        # 加载场景标签
        if scene_file and os.path.exists(scene_file):
            with open(scene_file, 'r') as f:
                self.scene_tags = json.load(f)
        
        super(CIPODataset, self).__init__(
            ann_file=ann_file,
            data_prefix=data_prefix,  # 🔧 修正
            **kwargs)
    
    # ...existing code...
    
    def _convert_cipo_to_coco(self, data_root, split='training'):
        """将CIPO格式转换为COCO格式"""
        
        data_dir = Path(data_root)
        ann_dir = data_dir / split
        img_dir = data_dir / "images" / split
        
        # COCO格式的基础结构
        coco_format = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "vehicle"},
                {"id": 2, "name": "pedestrian"}, 
                {"id": 3, "name": "sign"},
                {"id": 4, "name": "cyclist"}
            ]
        }
        
        # 类型映射
        type_mapping = {0: 1, 1: 1, 2: 2, 3: 3, 4: 4}  # CIPO type -> COCO category_id
        
        annotation_id = 1
        image_id = 1
        
        segment_dirs = [d for d in ann_dir.iterdir() if d.is_dir()]
        
        for segment_dir in segment_dirs:
            segment_name = segment_dir.name
            image_segment_dir = img_dir / segment_name
            
            if not image_segment_dir.exists():
                continue
            
            # 获取场景信息
            scene_info = self.scene_tags.get(segment_name, {})
            
            json_files = list(segment_dir.glob("*.json"))
            
            for json_file in json_files:
                # 对应的图像文件
                image_name = json_file.stem + ".jpg"
                image_path = image_segment_dir / image_name
                
                if not image_path.exists():
                    continue
                
                # 读取图像尺寸
                try:
                    img_info = mmcv.imread_info(str(image_path))
                    height, width = img_info['height'], img_info['width']
                except:
                    continue
                
                # 添加图像信息
                image_info = {
                    "id": image_id,
                    "file_name": f"{segment_name}/{image_name}",
                    "width": width,
                    "height": height,
                    "segment": segment_name,
                    "scene": scene_info.get('scene', 'unknown'),
                    "weather": scene_info.get('weather', 'unknown'),
                    "time": scene_info.get('hours', scene_info.get('time', 'unknown'))
                }
                coco_format["images"].append(image_info)
                
                # 读取标注
                with open(json_file, 'r') as f:
                    ann_data = json.load(f)
                
                objects = ann_data.get('result', ann_data.get('results', []))
                
                for obj in objects:
                    if 'id' not in obj or 'type' not in obj:
                        continue
                    
                    try:
                        cipo_level = int(obj['id'])
                        if not (1 <= cipo_level <= 4):
                            continue
                    except:
                        continue
                    
                    obj_type = obj.get('type', 0)
                    category_id = type_mapping.get(obj_type, 1)
                    
                    x = obj.get('x', 0)
                    y = obj.get('y', 0) 
                    width = obj.get('width', 0)
                    height = obj.get('height', 0)
                    
                    if width <= 0 or height <= 0:
                        continue
                    
                    # COCO格式的标注
                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x, y, width, height],  # COCO格式: [x, y, w, h]
                        "area": width * height,
                        "iscrowd": 0,
                        "cipo_level": cipo_level,  # 保留CIPO level信息
                        "scene": scene_info.get('scene', 'unknown'),
                        "weather": scene_info.get('weather', 'unknown'),
                        "time": scene_info.get('hours', scene_info.get('time', 'unknown'))
                    }
                    
                    coco_format["annotations"].append(annotation)
                    annotation_id += 1
                
                image_id += 1
        
        return coco_format
    
    def load_data_list(self):
        """重写数据加载方法，支持CIPO格式转换"""
        
        # 如果标注文件不存在，则进行转换
        if not os.path.exists(self.ann_file):
            print(f"Converting CIPO format to COCO format...")
            
            # 从路径推断数据根目录和split
            ann_file_path = Path(self.ann_file)
            data_root = ann_file_path.parent.parent
            split = ann_file_path.stem.replace('_coco', '')
            
            # 转换格式
            coco_data = self._convert_cipo_to_coco(data_root, split)
            
            # 保存转换后的文件
            os.makedirs(ann_file_path.parent, exist_ok=True)
            with open(self.ann_file, 'w') as f:
                json.dump(coco_data, f)
            
            print(f"Converted COCO file saved to: {self.ann_file}")
        
        # 调用父类方法加载COCO格式数据
        return super().load_data_list()
    
    def get_subset_by_scene(self, scenes):
        """根据场景过滤数据子集"""
        if isinstance(scenes, str):
            scenes = [scenes]
        
        filtered_data_list = []
        for data_info in self.data_list:
            img_info = data_info.get('img_info', {})
            if img_info.get('scene', 'unknown') in scenes:
                filtered_data_list.append(data_info)
        
        return filtered_data_list
    
    def get_subset_by_cipo_level(self, levels):
        """根据CIPO level过滤数据子集"""
        if isinstance(levels, int):
            levels = [levels]
        
        filtered_data_list = []
        for data_info in self.data_list:
            instances = data_info.get('instances', [])
            # 如果该图像包含指定level的目标，则保留
            if any(inst.get('cipo_level', 0) in levels for inst in instances):
                filtered_data_list.append(data_info)
        
        return filtered_data_list


# 数据转换工具
def convert_cipo_to_coco(data_root, output_dir, scene_file=None):
    """独立的转换工具函数"""
    
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 加载场景标签
    scene_tags = {}
    if scene_file and os.path.exists(scene_file):
        with open(scene_file, 'r') as f:
            scene_tags = json.load(f)
    
    for split in ['training', 'validation']:
        print(f"Converting {split} split...")
        
        dataset = CIPODataset.__new__(CIPODataset)
        dataset.scene_tags = scene_tags
        
        coco_data = dataset._convert_cipo_to_coco(data_root, split)
        
        output_file = output_dir / f"{split}_coco.json"
        with open(output_file, 'w') as f:
            json.dump(coco_data, f)
        
        print(f"Saved {split} annotations to: {output_file}")
        print(f"  Images: {len(coco_data['images'])}")
        print(f"  Annotations: {len(coco_data['annotations'])}")


if __name__ == '__main__':
    # 测试转换
    convert_cipo_to_coco(
        data_root='/Users/yigex/Documents/LLM-Inftra/OpenLane_CIPO/data/cipo',
        output_dir='/Users/yigex/Documents/LLM-Inftra/mmdetection/data/cipo',
        scene_file='/Users/yigex/Documents/LLM-Inftra/OpenLane_CIPO/data/cipo/scene.json'
    )