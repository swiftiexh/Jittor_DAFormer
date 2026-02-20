# Cityscapes 数据预处理

import argparse
import mmcv
import os.path as osp
from cityscapesscripts.preparation.json2labelImg import json2labelImg
import numpy as np
from PIL import Image
import json

# 命令行参数解析
# 运行指令：python utils/convert_datasets/cityscapes.py data/cityscapes --nproc 8
def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Cityscapes annotations to TrainIds')
    parser.add_argument('cityscapes_path', help='cityscapes data path')
    parser.add_argument('--gt-dir', default='gtFine', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args

# 转换函数：将 Cityscapes 标注文件转换为 TrainIds 格式
def convert_json_to_label(json_file):
    label_file = json_file.replace('_polygons.json', '_labelTrainIds.png')
    json2labelImg(json_file, label_file, 'trainIds')
    # 统计每个样本的类别像素数量，供 RCS 采样使用（只统计训练集样本）
    if 'train/' in json_file:
        pil_label = Image.open(label_file)
        label = np.asarray(pil_label)
        sample_class_stats = {}
        for c in range(19):
            n = int(np.sum(label == c))
            if n > 0:
                sample_class_stats[int(c)] = n
        sample_class_stats['file'] = label_file
        return sample_class_stats
    else:
        return None

# 保存所有样本的类别统计信息到 JSON 文件
def save_class_stats(out_dir, sample_class_stats):
    sample_class_stats = [e for e in sample_class_stats if e is not None]
    with open(osp.join(out_dir, 'sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(osp.join(out_dir, 'samples_with_class.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)

def main():
    args = parse_args()
    cityscapes_path = args.cityscapes_path
    out_dir = args.out_dir if args.out_dir else cityscapes_path
    mmcv.mkdir_or_exist(out_dir)
    gt_dir = osp.join(cityscapes_path, args.gt_dir)

    # 扫描 gtFine/ 目录，找到所有 *_polygons.json 文件，这些文件包含了标注信息
    poly_files = []
    for poly in mmcv.scandir(gt_dir, '_polygons.json', recursive=True):
        poly_file = osp.join(gt_dir, poly)
        poly_files.append(poly_file)
    
    if args.nproc > 1:
        sample_class_stats = mmcv.track_parallel_progress(
            convert_json_to_label, poly_files, args.nproc)
    else:
            sample_class_stats = mmcv.track_progress(convert_json_to_label,
                                                     poly_files)
    
    save_class_stats(out_dir, sample_class_stats)

if __name__ == '__main__':
    main()