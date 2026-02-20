# 图像变换

import os.path as osp
import numpy as np
from PIL import Image
import cv2
import jittor as jt

# 从文件加载图像
class LoadImageFromFile:
    def __call__(self, results):
        filename = osp.join(results['img_prefix'], 
                           results['img_info']['filename'])
        img = np.array(Image.open(filename).convert('RGB'))

        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['filename'] = results['img_info']['filename']
        return results

# 加载语义分割标注
class LoadAnnotations:
    def __call__(self, results):
        filename = osp.join(results['seg_prefix'],
                           results['img_info']['ann']['seg_map'])
        seg = np.array(Image.open(filename))
        results['gt_semantic_seg'] = seg
        results['seg_fields'].append('gt_semantic_seg')
        return results
    
# Resize
class Resize:
    def __init__(self, img_scale, keep_ratio=False):
        self.img_scale = img_scale
        self.keep_ratio = keep_ratio

    def __call__(self, results):
        img = results['img']

        if self.keep_ratio:
            # 保持比例缩放
            h, w = img.shape[:2]
            scale_h, scale_w = self.img_scale
            scale = min(scale_w / w, scale_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
        else:
            new_w, new_h = self.img_scale

        # 使用 cv2 缩放图像，图像缩放用线性插值
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        results['img'] = img
        results['img_shape'] = img.shape

        results['scale_factor'] = (new_w / results['ori_shape'][1], 
                                   new_h / results['ori_shape'][0])
        
        # 对标注进行相同的缩放，使用最近邻插值保持标签的离散性
        for key in results.get('seg_fields', []):
            seg = cv2.resize(results[key], (new_w, new_h), 
                           interpolation=cv2.INTER_NEAREST)
            results[key] = seg

        return results

# 随机裁剪图像和标注
class RandomCrop:
    def __init__(self, crop_size, cat_max_ratio=1.0, ignore_index=255):
        self.crop_size = crop_size  # (h, w)
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index
    
    # 随机生成裁剪框
    def get_crop_bbox(self, img, crop_size):
        h, w = img.shape[:2]
        crop_h, crop_w = crop_size
        # 如果图像小于裁剪尺寸，返回整个图像
        if h <= crop_h and w <= crop_w:
            return 0, 0, w, h
        # 随机选择左上角坐标
        margin_h = max(h - crop_h, 0)
        margin_w = max(w - crop_w, 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)

        crop_y1, crop_y2 = offset_h, min(offset_h + crop_h, h)
        crop_x1, crop_x2 = offset_w, min(offset_w + crop_w, w)

        return crop_x1, crop_y1, crop_x2, crop_y2
    
    # 执行裁剪
    def crop(self, img, crop_bbox):
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img
    
    def __call__(self, results):
        img = results['img']
        # 尝试找到满足 cat_max_ratio 的裁剪
        for _ in range(10):
            crop_bbox = self.get_crop_bbox(img, self.crop_size)
            # 检查类别最大比例
            if self.cat_max_ratio < 1.0:
                # 裁剪标注并检查
                seg_temp = results['gt_semantic_seg']
                seg_temp = self.crop(seg_temp, crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) > self.cat_max_ratio:
                    continue
            # 找到合适的裁剪框
            break
        # 裁剪图像
        results['img'] = self.crop(img, crop_bbox)
        results['img_shape'] = results['img'].shape
        # 裁剪标注
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)
        return results
    
# 随机水平翻转
class RandomFlip:
    def __init__(self, prob=0.5, direction='horizontal'):
        self.prob = prob
        self.direction = direction
        assert direction in ['horizontal', 'vertical']

    def __call__(self, results):
        if np.random.rand() < self.prob:
            # 翻转图像
            img = results['img']
            if self.direction == 'horizontal':
                results['img'] = np.flip(img, axis=1).copy()
            else:
                results['img'] = np.flip(img, axis=0).copy() 
            # 翻转标注
            for key in results.get('seg_fields', []):
                if self.direction == 'horizontal':
                    results[key] = np.flip(results[key], axis=1).copy()
                else:
                    results[key] = np.flip(results[key], axis=0).copy() 
            results['flip'] = True
            results['flip_direction'] = self.direction
        else:
            results['flip'] = False
            results['flip_direction'] = None
        return results
    
# 归一化图像
class Normalize:
    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        img = results['img'].astype(np.float32)
        # 归一化
        # 图像已经是 RGB 格式（由 LoadImageFromFile 保证），直接归一化
        img = (img - self.mean) / self.std
        results['img'] = img
        results['img_norm_cfg'] = dict(
            mean=self.mean.tolist(),
            std=self.std.tolist(),
            to_rgb=self.to_rgb
        )
        return results
    
# 填充图像和标注到指定大小
class Pad:
    def __init__(self, size=None, pad_val=0, seg_pad_val=255):
        self.size = size
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

    def __call__(self, results):
        img = results['img']
        if self.size is not None:
            # 填充到指定大小
            pad_h, pad_w = self.size
        else:
            return results
        # 计算填充量
        h, w = img.shape[:2]
        pad_top = 0
        pad_left = 0
        pad_bottom = max(pad_h - h, 0)
        pad_right = max(pad_w - w, 0)
        # 填充图像
        img = cv2.copyMakeBorder(
            img, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=self.pad_val
        )
        results['img'] = img
        results['pad_shape'] = img.shape
        # 填充标注
        for key in results.get('seg_fields', []):
            seg = cv2.copyMakeBorder(
                results[key], pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=self.seg_pad_val
            )
            results[key] = seg
        return results

# 默认格式打包：图像 HWC->CHW 转 Var，标注增加维度转 Var
class DefaultFormatBundle:
    def __call__(self, results):
        # 处理图像：HWC -> CHW -> Var
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1) # 在最后增加一个通道维度
            # 转置并确保连续内存
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = jt.array(img)
        # 处理标注：增加维度 -> Var
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = jt.array(
                results['gt_semantic_seg'][None, ...].astype(np.int64)
            )
        return results

# 收集数据，返回指定键的数据
class Collect:
    def __init__(self, keys):
        self.keys = keys
    
    def __call__(self, results):
        data = {}
        for key in self.keys:
            data[key] = results[key]
        return data