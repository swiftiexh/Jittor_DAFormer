# 类别混合增强

import jittor as jt
import numpy as np

# 生成指定类别的掩码
# 每个位置上,判断 label 是否属于 classes 中的某个类别
def generate_class_mask(label, classes):
    # label:[1,H,W] classes: [nclasses/2]
    # 扩展维度用于广播 迁移自 broadcast_tensors
    label_expanded = label.unsqueeze(0)  # [1,1,H,W]
    classes_expanded = classes.reshape(-1, 1, 1, 1)  # [nclasses/2,1,1,1]
    
    # 广播比较:label_expanded 与每个类别比较
    # [1,1,H,W] == [nclasses/2,1,1,1] -> [nclasses/2,1,H,W]
    class_mask = (label_expanded == classes_expanded).sum(dim=0, keepdims=True)  # [1,1,H,W]
    class_mask = (class_mask > 0).float32()  # 转为 0/1 掩码
    return class_mask.squeeze(0)  # [1,H,W]

# 为每个样本生成掩码
def get_class_masks(labels):
    class_masks = []
    # labels: [B,1,H,W]
    for label in labels:
        # label: [1,H,W]
        # 获取图像中出现的类别
        unique_classes = jt.unique(label)  # [nclasses]
        nclasses = unique_classes.shape[0]  # 类别数量
        
        # 随机选择一半的类别 (向上取整)
        class_choice = np.random.choice(
            nclasses, 
            int((nclasses + nclasses % 2) / 2),  # 选择一半,奇数时向上取整
            replace=False
        )
        classes = unique_classes[jt.array(class_choice).int64()]

        # 生成类别掩码
        class_mask = generate_class_mask(label, classes).unsqueeze(0)  # [1,1,H,W]
        class_masks.append(class_mask)

    return class_masks  # List of [B tensors of shape [1,1,H,W]]

# 进行一次类别混合
def one_mix(mask, data=None, target=None):
    if mask is None:
        return data, target
    # mask: [1,1,H,W] -> 取出 [H,W]
    mask_2d = mask[0, 0]  # [H,W]
    if data is not None:
        # data: [2,3,H,W]
        mask_3d = mask_2d.unsqueeze(0)  # [1,H,W]
        data = (mask_3d * data[0] + (1 - mask_3d) * data[1]).unsqueeze(0)  # [1,3,H,W]
    if target is not None:
        # target: [2,H,W]
        target = (mask_2d * target[0] + (1 - mask_2d) * target[1]).unsqueeze(0)  # [1,H,W]
    return data, target

# 颜色抖动函数 
# Jittor 有 ColorJitter 但是只支持 PIL 图像
def color_jitter(color_jitter, mean, std, data=None, target=None, s=0.25, p=0.2):
    if data is None:
        return data, target
    if color_jitter < p:  # 以概率 p 进行颜色抖动 (color_jitter 是 [0,1] 随机数)
        # 反归一化到 [0, 1] (期望 mean/std 维度为 [1,3,1,1])
        data = data * std + mean
        data = data / 255.0
        # 按照 torchvision.transforms.ColorJitter 的实现顺序和逻辑
        # 1. Brightness: 亮度调整 img * brightness_factor
        brightness_factor = np.random.uniform(max(0, 1 - s), 1 + s)
        data = data * brightness_factor
        # 2. Contrast: 对比度调整 (img - mean) * contrast_factor + mean
        contrast_factor = np.random.uniform(max(0, 1 - s), 1 + s)
        # 计算每个样本每个通道的均值
        mean_val = data.mean(dims=(2, 3), keepdims=True)  # [B,C,1,1]
        data = (data - mean_val) * contrast_factor + mean_val
        # 3. Saturation: 饱和度调整 (转灰度图后插值)
        sat_factor = np.random.uniform(max(0, 1 - s), 1 + s)
        # RGB to Grayscale: 0.2989*R + 0.5870*G + 0.1140*B (ITU-R BT.601标准)
        gray = (data[:, 0:1, :, :] * 0.2989 + 
                data[:, 1:2, :, :] * 0.5870 + 
                data[:, 2:3, :, :] * 0.1140)  # [B,1,H,W]
        data = gray + (data - gray) * sat_factor
        # 裁剪到 [0, 1]
        data = data.safe_clip(0.0, 1.0)
        # 重新归一化
        data = data * 255.0
        data = (data - mean) / std
    return data, target

# 高斯模糊函数
# 使用 depthwise convolution 高效实现 (一次卷积处理所有通道)
def gaussian_blur(blur, data=None, target=None):
    if data is None:
        return data, target
    if blur > 0.5:
        # 随机选择 sigma
        sigma = np.random.uniform(0.15, 1.15)
        # 基于 sigma 计算 kernel size (torchvision 标准方式)
        # 3-sigma 规则：99.7% 的高斯分布值在 ±3σ 范围内
        kernel_size = max(3, 2 * int(3 * sigma) + 1)  # 至少 3×3, 确保为奇数
        # 创建 1D 高斯核
        def get_gaussian_kernel_1d(kernel_size, sigma):
            x = jt.arange(kernel_size).float32() - (kernel_size - 1) / 2
            gauss = jt.exp(-x.pow(2) / (2 * sigma ** 2))
            return gauss / gauss.sum()
        # 生成 2D 高斯核 (可分离卷积：1D 核外积)
        kernel_1d = get_gaussian_kernel_1d(kernel_size, sigma)
        kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)  # [K, K]
        # 扩展为 depthwise 卷积核 [C, 1, K, K]
        B, C, H, W = data.shape
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
        kernel_depthwise = kernel_2d.repeat(C, 1, 1, 1)  # [C, 1, K, K]
        # 使用 depthwise convolution (groups=C) 一次性处理所有通道
        padding = kernel_size // 2
        data = jt.nn.conv2d(data, kernel_depthwise, groups=C, padding=padding)
    return data, target

# 强增强变换函数
def strong_transform(param, data=None, target=None):
    # 1. 类别混合
    data, target = one_mix(mask=param['mix'], data=data, target=target)
    # 2. 颜色抖动
    data, target = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data,
        target=target
    )
    # 3. 高斯模糊
    data, target = gaussian_blur(blur=param['blur'], data=data, target=target)
    return data, target

# 混合增强函数
def apply_class_mix(src_img, src_seg, tgt_img, pseudo_label, 
                   pseudo_weight, mix_type, blur, 
                   color_jitter_s, color_jitter_p):
    assert mix_type == 'class'
    batch_size = src_img.shape[0]

    # 1. 生成类别掩码 (每个样本随机选择一半类别)
    mix_masks = get_class_masks(src_seg) # List of [B tensors of shape [1,1,H,W]]

    # 2. 对每个样本进行混合
    mixed_img_list = []
    mixed_lbl_list = []
    mixed_weight_list = []
    # 准备归一化参数 (同训练时的 norm_cfg)
    mean = jt.array([123.675, 116.28, 103.53]).reshape(1, 3, 1, 1)
    std = jt.array([58.395, 57.12, 57.375]).reshape(1, 3, 1, 1)
    
    for i in range(batch_size):
        strong_parameters = {
            'mix': mix_masks[i],  # [1,1,H,W]
            'color_jitter': np.random.uniform(0, 1),  # 随机颜色扰动强度
            'color_jitter_s': color_jitter_s,  # 颜色扰动强度上限
            'color_jitter_p': color_jitter_p,  # 颜色扰动概率
            'blur': np.random.uniform(0, 1) if blur else 0.0,  # 随机模糊强度
            'mean': mean,  # [1,3,1,1]
            'std': std     # [1,3,1,1]
        }
        # 混合图像和标签
        mixed_img_i, mixed_lbl_i = strong_transform(
            strong_parameters,
            data=jt.stack([src_img[i], tgt_img[i]]),  # [2,3,H,W]
            target=jt.stack([src_seg[i][0], pseudo_label[i]])  # [2,H,W]
        )  # each: [1,3,H,W], [1,H,W]
        
        # 混合权重
        # 为源域 GT 创建全1权重
        gt_pixel_weight = jt.ones_like(pseudo_weight[i])
        _, mixed_weight_i = strong_transform(
            strong_parameters,
            target=jt.stack([gt_pixel_weight, pseudo_weight[i]])  # [2,H,W]
        )  # each: [1,H,W]
        
        # 收集结果
        mixed_img_list.append(mixed_img_i)  # List of [B tensors of shape [1,3,H,W]]
        mixed_lbl_list.append(mixed_lbl_i)  # List of [B tensors of shape [1,H,W]]
        mixed_weight_list.append(mixed_weight_i)  # List of [B tensors of shape [1,H,W]]
    
    # 3. 拼接为 batch
    mixed_img = jt.concat(mixed_img_list, dim=0)  # [B,3,H,W]
    mixed_lbl = jt.concat(mixed_lbl_list, dim=0)  # [B,H,W]
    mixed_weight = jt.concat(mixed_weight_list, dim=0)  # [B,H,W]
    
    return mixed_img, mixed_lbl, mixed_weight

