# GTA 数据预处理

import argparse
import mmcv
import os.path as osp
from PIL import Image
import numpy as np
import json

# 命令行参数解析
# 运行指令：python utils/convert_datasets/gta.py data/gta --nproc 8
def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert GTA annotations to TrainIds')
    parser.add_argument('gta_path', help='gta data path')
    parser.add_argument('--gt-dir', default='labels', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=4, type=int, help='number of process')
    args = parser.parse_args()
    return args

# 转换函数：将 GTA 标注文件转换为 TrainIds 格式
def convert_to_train_id(file):
    pil_label = Image.open(file)
    label = np.asarray(pil_label)
    id_to_trainid = {
        7: 0,
        8: 1,
        11: 2,
        12: 3,
        13: 4,
        17: 5,
        19: 6,
        20: 7,
        21: 8,
        22: 9,
        23: 10,
        24: 11,
        25: 12,
        26: 13,
        27: 14,
        28: 15,
        31: 16,
        32: 17,
        33: 18
    }
    label_copy = 255 * np.ones(label.shape, dtype=np.uint8) # 初始化为 255（忽略索引）
    sample_class_stats = {} # 存储每个样本的类别统计信息
    for k, v in id_to_trainid.items():
        k_mask = label == k # k_mask 是一个布尔数组，表示哪些像素属于类别 k
        label_copy[k_mask] = v # 将类别 k 的像素值替换为对应的 TrainId v
        n = int(np.sum(k_mask)) # 计算类别 k 的像素数量
        if n > 0:
            sample_class_stats[v] = n # 记录类别 v 的像素数量
    new_file = file.replace('.png', '_labelTrainIds.png')
    assert file != new_file
    sample_class_stats['file'] = new_file
    Image.fromarray(label_copy, mode='L').save(new_file)
    return sample_class_stats

# 保存类别统计信息到 JSON 文件
def save_class_stats(out_dir, sample_class_stats):
    # 列表格式，每个元素是一张图的类别统计
    with open(osp.join(out_dir, 'sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2) 

    # 字典格式，键是文件名，值是类别统计
    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)
    
    # 反向索引：哪些文件包含某个类别
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
    args = parse_args() # 解析命令行参数
    gta_path = args.gta_path
    out_dir = args.out_dir if args.out_dir else gta_path
    mmcv.mkdir_or_exist(out_dir)
    gt_dir = osp.join(gta_path, args.gt_dir) # GTA 标注目录

    # 扫描标注目录，获取所有标注文件路径
    poly_files = [] # 存储所有标注文件路径
    for poly in mmcv.scandir(
            gt_dir, suffix=tuple(f'{i}.png' for i in range(10)),
            recursive=True):
        poly_file = osp.join(gt_dir, poly)
        poly_files.append(poly_file)
    poly_files = sorted(poly_files)

    # 定义转换函数，将标注文件转换为 TrainIds 格式 
    if args.nproc > 1:
        sample_class_stats = mmcv.track_parallel_progress(
                convert_to_train_id, poly_files, args.nproc)
    else:
        sample_class_stats = mmcv.track_progress(convert_to_train_id,
                                                 poly_files)
        
    save_class_stats(out_dir, sample_class_stats)
        
    
if __name__ == '__main__':
    main()