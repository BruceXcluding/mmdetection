"""
CIPO训练脚本 - 基于MMDetection
使用方法：
python projects/cipo/train_cipo.py projects/cipo/configs/rtmdet_l_8xb32-300e_cipo.py
"""

import os
import sys
import argparse
from pathlib import Path

# 🔧 添加项目路径到sys.path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

def main():
    parser = argparse.ArgumentParser(description='Train CIPO models with MMDetection')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume', help='resume from the latest checkpoint')
    parser.add_argument('--amp', action='store_true', help='enable automatic-mixed-precision training')
    parser.add_argument('--auto-scale-lr', action='store_true',
                       help='enable automatically scaling LR.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], 
                       default='none', help='job launcher')
    
    args = parser.parse_args()
    
    # 确保数据已经转换
    data_root = '/Users/yigex/Documents/LLM-Inftra/OpenLane_CIPO/data/cipo'
    mmdet_data_dir = Path('./data/cipo')
    
    if not (mmdet_data_dir / 'annotations').exists():
        print("Converting CIPO data to COCO format...")
        from cipo_dataset import convert_cipo_to_coco  # 🔧 修正导入
        
        convert_cipo_to_coco(
            data_root=data_root,
            output_dir=mmdet_data_dir,
            scene_file=f'{data_root}/scene.json'
        )
        
        # 创建软链接指向图像目录
        images_link = mmdet_data_dir / 'images'
        if not images_link.exists():
            images_source = Path(data_root) / 'images'
            if images_source.exists():
                images_link.symlink_to(images_source)  # 🔧 修正symlink方法
                print(f"Created symlink: {images_link} -> {images_source}")
    
    # 构建训练命令
    cmd_parts = [
        'python tools/train.py',
        args.config
    ]
    
    if args.work_dir:
        cmd_parts.append(f'--work-dir {args.work_dir}')
    
    if args.resume:
        cmd_parts.append(f'--resume {args.resume}')
    
    if args.amp:
        cmd_parts.append('--amp')
    
    if args.auto_scale_lr:
        cmd_parts.append('--auto-scale-lr')
    
    if args.launcher != 'none':
        cmd_parts.append(f'--launcher {args.launcher}')
    
    cmd = ' '.join(cmd_parts)
    print(f"Running command: {cmd}")
    
    # 执行训练
    os.system(cmd)

if __name__ == '__main__':
    main()