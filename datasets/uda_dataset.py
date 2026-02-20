# 核心是组合 source 和 target，支持 RCS

import json
import os.path as osp
import jittor as jt
from jittor.dataset import Dataset
import numpy as np

# 计算稀有类采样的类别概率
def get_rcs_class_probs(data_root, temperature):
    # 读取反向索引：哪些文件包含某个类别
    with open(osp.join(data_root, 'samples_with_class.json'), 'r') as f:
        samples_with_class = json.load(f)
    
    # 统计每个类别的总像素数
    overall_class_stats = {}
    for class_id, file_list in samples_with_class.items():
        class_id = int(class_id)
        total_pixels = sum(n for _, n in file_list)  # file_list: [(file, n), ...]
        overall_class_stats[class_id] = total_pixels

    # 按像素数排序（从少到多）
    overall_class_stats = {
        k: v for k, v in sorted(
            overall_class_stats.items(), key=lambda item: item[1]
        )
    }

    classes_list = list(overall_class_stats.keys())
    pixels_list = list(overall_class_stats.values())
    # 使用 numpy float64 避免精度丢失
    freq = np.array(pixels_list, dtype=np.float64)
    freq = freq / freq.sum()
    freq = 1.0 - freq
    # 转回 Jittor 做 softmax
    freq = jt.array(freq).float32()
    freq = jt.nn.softmax(freq / temperature, dim=-1)
    return classes_list, freq.numpy()

class UDADataset(Dataset):
    def __init__(self, source, target, rare_class_sampling=None, batch_size=2, num_workers=4, shuffle=True):
        super().__init__()
        self.source = source
        self.target = target
        self.rcs_cfg = rare_class_sampling
        self.rcs_enabled = rare_class_sampling is not None

        self.ignore_index = target.ignore_index
        self.CLASSES = target.CLASSES
        self.PALETTE = target.PALETTE

        # 设置 Jittor DataLoader 属性
        self.set_attrs(batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

        # 初始化 RCS
        if self.rcs_cfg:
            self._init_rcs()
    
    def __len__(self):
        # 返回数据集长度（用于 DataLoader）
        if self.rcs_enabled:
            # RCS 模式下，返回源域长度（因为每次采样都是从源域选择）
            return len(self.source)
        else:
            # 简单配对模式：返回源域和目标域的最大长度
            return max(len(self.source), len(self.target))
    
    # 初始化稀有类采样
    def _init_rcs(self):
        self.rcs_class_temp = self.rcs_cfg['class_temp']
        self.rcs_min_crop_ratio = self.rcs_cfg['min_crop_ratio']
        self.rcs_min_pixels = self.rcs_cfg['min_pixels']

        # 计算类别采样概率
        self.rcs_classes, self.rcs_classprob = get_rcs_class_probs(
            self.source.data_root, self.rcs_class_temp
        ) # 类别 ID 列表（python 列表） 和 对应采样概率（numpy 数组）

        # 读取每个类别对应的样本文件列表
        samples_with_class_path = osp.join(
            self.source.data_root, 'samples_with_class.json'
        )
        with open(samples_with_class_path, 'r') as f:
            samples_with_class_and_n = json.load(f)

        # 过滤：只保留 像素数 > min_pixels 的样本
        # 构建从类别到文件列表的映射
        self.samples_with_class = {}
        for class_id in self.rcs_classes:
            self.samples_with_class[str(class_id)] = []
            for file_path, num_pixels in samples_with_class_and_n[str(class_id)]:
                if num_pixels > self.rcs_min_pixels:
                    # 只保留文件名（去掉路径）
                    filename = file_path.split('/')[-1]
                    self.samples_with_class[str(class_id)].append(filename)
        
        # 构建从文件名到源域数据集索引的映射
        self.file_to_idx = {}
        for i, img_info in enumerate(self.source.img_infos):
            seg_map_file = img_info['ann']['seg_map']
            # 提取文件名（兼容不同路径格式）
            filename = seg_map_file.split('/')[-1]
            self.file_to_idx[filename] = i

    def get_rare_class_sample(self):
        # 1. 按概率采样一个类别
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        # 2. 从该类别的样本中随机选一个文件
        filename = np.random.choice(self.samples_with_class[str(c)])
        source_idx = self.file_to_idx[filename]
        # 3. 获取源域样本（会触发 pipeline，包含 RandomCrop）
        source_sample = self.source[source_idx]
        # 4. 检查裁剪后该类别的像素数是否满足 min_crop_ratio
        if self.rcs_min_crop_ratio > 0:
            for attempt in range(10):
                # 统计当前裁剪中类别 c 的像素数
                gt_seg = source_sample['gt_semantic_seg']
                # 处理可能的 Var 或 numpy 格式
                if isinstance(gt_seg, jt.Var):
                    n_class = jt.sum(gt_seg == c).item()
                else:
                    n_class = np.sum(gt_seg == c)
                                # 如果满足最小比例要求，跳出循环
                if n_class > self.rcs_min_pixels * self.rcs_min_crop_ratio:
                    break
                # 否则重新采样（RandomCrop 会生成新的裁剪）
                source_sample = self.source[source_idx]
        # 5. 随机选一个目标域样本
        target_idx = np.random.randint(0, len(self.target))
        target_sample = self.target[target_idx]
        # 6. 合并源域和目标域数据
        return {
            **source_sample,  # 源域的 img, gt_semantic_seg
            'target_img': target_sample['img']
        }
    
    def __getitem__(self, idx):
        if self.rcs_enabled:
            # RCS 模式：忽略 idx，按稀有类概率采样
            return self.get_rare_class_sample()
        else:
            # 简单配对模式：源域和目标域循环配对
            source_idx = idx % len(self.source)
            target_idx = idx % len(self.target)
            source_sample = self.source[source_idx]
            target_sample = self.target[target_idx]
            return {
                **source_sample,
                'target_img': target_sample['img']
            }