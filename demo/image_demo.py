# 单张图片推理脚本

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import argparse
import numpy as np
from PIL import Image
import jittor as jt

CITYSCAPES_PALETTE = [
    (128, 64, 128),  # 0: road
    (244, 35, 232),  # 1: sidewalk
    (70, 70, 70),    # 2: building
    (102, 102, 156), # 3: wall
    (190, 153, 153), # 4: fence
    (153, 153, 153), # 5: pole
    (250, 170, 30),  # 6: traffic light
    (220, 220, 0),   # 7: traffic sign
    (107, 142, 35),  # 8: vegetation
    (152, 251, 152), # 9: terrain
    (70, 130, 180),  # 10: sky
    (220, 20, 60),   # 11: person
    (255, 0, 0),     # 12: rider
    (0, 0, 142),     # 13: car
    (0, 0, 70),      # 14: truck
    (0, 60, 100),    # 15: bus
    (0, 80, 100),    # 16: train
    (0, 0, 230),     # 17: motorcycle
    (119, 11, 32)    # 18: bicycle
]

CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle'
]


def parse_args():
    parser = argparse.ArgumentParser(description='Single image inference demo')
    parser.add_argument('image', help='path to input image')
    parser.add_argument('checkpoint', help='path to checkpoint file')
    parser.add_argument('--config', default='configs/gta2cs_daformer.py',
                        help='path to config file (default: configs/gta2cs_daformer.py)')
    parser.add_argument('--out-dir', default='demo',
                        help='output directory (default: demo/output)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='device to use (default: cuda)')
    return parser.parse_args()

# 加载并预处理输入图片
def load_image(image_path):
    # 加载图片
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    # 保存原始尺寸
    ori_shape = img_array.shape
    # 保持宽高比resize，长边对齐到1024
    h, w = ori_shape[0], ori_shape[1]
    if h > w:
        new_h = 1024
        new_w = int(w * 1024 / h)
    else:
        new_w = 1024
        new_h = int(h * 1024 / w)
    img_resized = np.array(Image.fromarray(img_array).resize((new_w, new_h), Image.BILINEAR))
    # 归一化
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    img_normalized = (img_resized - mean) / std
    # HWC -> CHW
    img_chw = img_normalized.transpose(2, 0, 1)
    # 转为Jittor Var并添加batch维度 [1, 3, H, W]
    img_tensor = jt.array(img_chw).float32().unsqueeze(0)
    return img_tensor, ori_shape


# 将类别ID掩码转换为彩色图像
def colorize_mask(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for train_id, color in enumerate(CITYSCAPES_PALETTE):
        color_mask[mask == train_id] = color
    return color_mask

# 推理函数
def inference(model, img_tensor, ori_shape):
    model.eval()
    with jt.no_grad():
        # 前向传播
        seg_logits = model.encode_decode(img_tensor)  # [1, 19, H, W]
        
        # 获取预测类别
        seg_pred = seg_logits.argmax(dim=1)  # [1, H, W]
        
        # 转为numpy并去除所有单维度
        seg_pred_np = seg_pred.numpy().astype(np.uint8)
        seg_pred_np = np.squeeze(seg_pred_np)  # 去除所有大小为1的维度
        
        # 确保是2D
        if seg_pred_np.ndim != 2:
            raise ValueError(f"Expected 2D array after squeeze, got shape {seg_pred_np.shape}")
        
        print(f"  Segmentation mask shape before resize: {seg_pred_np.shape}")
        
        # Resize回原始尺寸
        seg_pred_resized = np.array(
            Image.fromarray(seg_pred_np, mode='L').resize(  # 明确指定mode='L'（灰度图）
                (ori_shape[1], ori_shape[0]), 
                Image.NEAREST
            )
        )
    
    return seg_pred_resized


def main():
    args = parse_args()
    # 设置设备
    if args.device == 'cuda':
        if jt.has_cuda:
            jt.flags.use_cuda = 1
            print("Using CUDA (GPU)")
        else:
            print("CUDA not available, using CPU")
            jt.flags.use_cuda = 0
    else:
        jt.flags.use_cuda = 0
        print("Using CPU")
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    # 1. 加载配置
    print(f"Loading config from {args.config}...")
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from train import load_config, build_model
    cfg = load_config(args.config)
    # 2. 构建模型
    print("Building model...")
    model = build_model(cfg.model)
    # 3. 加载checkpoint（仅加载模型权重）
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = jt.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'])
    print("Model weights loaded")
    # 4. 加载图片
    print(f"Loading image from {args.image}...")
    img_tensor, ori_shape = load_image(args.image)
    print(f"  Original shape: {ori_shape}")
    print(f"  Input tensor shape: {img_tensor.shape}")
    # 5. 推理
    print("Running inference...")
    seg_pred = inference(model, img_tensor, ori_shape)
    print(f"  Prediction shape: {seg_pred.shape}")
    # 6. 生成彩色分割结果
    print("Generating segmentation result...")
    color_mask = colorize_mask(seg_pred)
    # 7. 保存结果
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    output_path = os.path.join(args.out_dir, f"{base_name}_seg.png")
    Image.fromarray(color_mask).save(output_path)
    print(f"Saved to: {output_path}")
    print("\n Inference completed!")


if __name__ == '__main__':
    main()