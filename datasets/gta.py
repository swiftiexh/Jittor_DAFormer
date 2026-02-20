# GTA 数据集类

from .base import BaseDataset
import os

class GTADataset(BaseDataset):
    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
            'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
            'bicycle')
    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]
    
    def __init__(self, data_root, img_dir, ann_dir, pipeline, batch_size=1, num_workers=0, shuffle=False):
        super().__init__(
            data_root=data_root,
            img_dir=img_dir,
            ann_dir=ann_dir,
            img_suffix='.png',
            seg_map_suffix='_labelTrainIds.png',
            pipeline=pipeline,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle
        )
    
    # 扫描 images/ 目录，匹配 labels/ 里的标注
    def load_annotations(self):
        img_infos = []
        for filename in sorted(os.listdir(self.img_dir)):
            if filename.endswith(self.img_suffix):
                ann_file = filename.replace(self.img_suffix, self.seg_map_suffix)
                img_infos.append(dict(
                    filename=filename, # 比如： 000000_000019.png
                    ann=dict(seg_map=ann_file) # 对应标注路径
                ))
        return img_infos