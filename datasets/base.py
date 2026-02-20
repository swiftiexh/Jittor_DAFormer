# 基础数据集类

import os.path as osp
import jittor as jt
from jittor.dataset import Dataset

class BaseDataset(Dataset):
    CLASSES = None # 数据集类别
    PALETTE = None # 颜色调色板

    # 初始化数据集
    def __init__(self, data_root, img_dir, ann_dir, 
                img_suffix='.png', seg_map_suffix='.png',
                pipeline=None, batch_size=1, num_workers=0, shuffle=False):
        super().__init__()
        self.data_root = data_root
        self.img_dir = osp.join(data_root, img_dir)
        self.ann_dir = osp.join(data_root, ann_dir)
        self.img_suffix = img_suffix # 图像文件后缀
        self.seg_map_suffix = seg_map_suffix # 标注文件后缀
        self.pipeline = pipeline or []
        self.ignore_index = 255 # 忽略索引

        # 加载图像和标注信息
        self.img_infos = self.load_annotations()
        
        # 设置 Jittor DataLoader 属性
        self.set_attrs(batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    # 扫描数据目录，返回 img_infos 列表
    def load_annotations(self):
        raise NotImplementedError
    
    # 返回数据集长度
    def __len__(self):
        return len(self.img_infos)
    
    # 根据索引获取数据样本
    def __getitem__(self, idx):
        img_info = self.img_infos[idx]
        results = dict(
            img_info=img_info,
            img_prefix=self.img_dir,
            seg_prefix=self.ann_dir,
            seg_fields=[] # 标注字段列表，有的任务可能有多个标注字段，如实例掩码等等
        )
        for transform in self.pipeline:
            results = transform(results)
            if results is None:
                return None
        return results