# 伪标签生成

import jittor as jt
import numpy as np

def generate_pseudo_label(logits, threshold, ignore_top, ignore_bottom):
    # 1. Softmax 得到概率分布
    ema_softmax = jt.nn.softmax(logits.stop_grad(), dim=1)  # [B, C, H, W]

    # 2. 取最大概率及其对应的类别
    pseudo_prob = jt.max(ema_softmax,dim=1,keepdims=False) # [B, H, W]
    pseudo_label = jt.argmax(ema_softmax,dim=1,keepdims=False)[0] # [B, H, W]

    # 3. 生成置信度掩码: 概率 >= threshold 的像素
    ps_large_p = (pseudo_prob >= threshold)  # [B, H, W], bool

    # 4. 计算高置信度像素的比例 (用于统计)
    ps_size = pseudo_label.numpy().size  # 总像素数
    pseudo_ratio = ps_large_p.sum().data[0] / ps_size

    # 5. 初始化权重: 所有高置信度像素权重为 pseudo_ratio
    # 这里权重值是统一的 pseudo_ratio,而不是每个像素的实际概率
    pseudo_weight = pseudo_ratio * jt.ones_like(pseudo_prob) # [B, H, W]

    # 6. 将低置信度像素的权重设为 0
    pseudo_weight = jt.where(ps_large_p, pseudo_weight, jt.zeros_like(pseudo_weight))

    # 7. 忽略顶部像素 (天空、建筑顶部等容易有伪影)
    if ignore_top > 0:
        pseudo_weight[:, :ignore_top, :] = 0
    
    # 8. 忽略底部像素 (车辆引擎盖、道路底部等容易有伪影)
    if ignore_bottom > 0:
        pseudo_weight[:, -ignore_bottom:, :] = 0
    
    return pseudo_label, pseudo_weight
