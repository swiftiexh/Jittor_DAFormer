# 绘制训练日志曲线

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def load_jsonl(file_path):
    logs = []
    if not os.path.exists(file_path):
        return logs
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                logs.append(json.loads(line))
    return logs

# 绘制训练曲线
def plot_training_curves(log_dir, save_dir=None):
    train_log_path = os.path.join(log_dir, 'train.json')
    val_log_path = os.path.join(log_dir, 'val.json')
    
    # 加载日志
    train_logs = load_jsonl(train_log_path)
    val_logs = load_jsonl(val_log_path)
    
    # 过滤出实际的训练和验证记录
    train_logs = [log for log in train_logs if log.get('type') == 'train']
    val_logs = [log for log in val_logs if log.get('type') == 'validation']
    
    if not train_logs:
        print("No training logs found!")
        return
    
    # 提取数据
    iterations = [log['iteration'] for log in train_logs]
    
    # 创建图表 - 6个主要训练指标
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Curves', fontsize=16)
    
    # 1. Total Loss
    if 'total_loss' in train_logs[0]:
        total_loss = [log['total_loss'] for log in train_logs]
        axes[0, 0].plot(iterations, total_loss, 'b-', linewidth=1, alpha=0.7)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Source Loss
    if 'src_loss' in train_logs[0]:
        src_loss = [log['src_loss'] for log in train_logs]
        axes[0, 1].plot(iterations, src_loss, 'r-', linewidth=1, alpha=0.7)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Source Loss')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Mix Loss
    if 'mix_loss' in train_logs[0]:
        mix_loss = [log['mix_loss'] for log in train_logs]
        axes[0, 2].plot(iterations, mix_loss, 'g-', linewidth=1, alpha=0.7)
        axes[0, 2].set_xlabel('Iteration')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].set_title('Mix Loss')
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Feature Distance Loss
    if 'loss_imnet_feat_dist' in train_logs[0]:
        fdist_loss = [log.get('loss_imnet_feat_dist', 0) for log in train_logs]
        axes[1, 0].plot(iterations, fdist_loss, 'm-', linewidth=1, alpha=0.7)
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Feature Distance Loss')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Learning Rate
    if 'lr' in train_logs[0]:
        lr = [log['lr'] for log in train_logs]
        axes[1, 1].plot(iterations, lr, 'b-', linewidth=1, alpha=0.7)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Pseudo Ratio
    if 'pseudo_ratio' in train_logs[0]:
        pseudo_ratio = [log['pseudo_ratio'] for log in train_logs]
        axes[1, 2].plot(iterations, pseudo_ratio, 'orange', linewidth=1, alpha=0.7)
        axes[1, 2].set_xlabel('Iteration')
        axes[1, 2].set_ylabel('Pseudo Ratio')
        axes[1, 2].set_title('Pseudo Ratio')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存或显示
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()

# 绘制每类IoU柱状图
def plot_class_iou(log_dir, save_dir=None):
    val_log_path = os.path.join(log_dir, 'val.json')
    val_logs = load_jsonl(val_log_path)
    val_logs = [log for log in val_logs if log.get('type') == 'validation']
    
    if not val_logs:
        print("No validation logs found!")
        return
    
    # 获取最后一次验证的结果
    last_val = val_logs[-1]
    
    # Cityscapes 19类别名称
    class_names = [
        'road', 'sidewalk', 'building', 'wall', 'fence',
        'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
        'sky', 'person', 'rider', 'car', 'truck',
        'bus', 'train', 'motorcycle', 'bicycle'
    ]
    
    # 提取每类IoU（直接使用类别名称作为key）
    class_ious = []
    for class_name in class_names:
        if class_name in last_val:
            class_ious.append(last_val[class_name])
    
    if not class_ious:
        print("No class IoU data found!")
        return
    
    # 绘制柱状图
    plt.figure(figsize=(15, 6))
    x = np.arange(len(class_ious))
    plt.bar(x, class_ious, color='steelblue', alpha=0.8)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('IoU', fontsize=12)
    plt.title(f'Per-Class IoU (Iteration {last_val["iteration"]})', fontsize=14)
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # 添加mIoU线
    miou = last_val['mIoU']
    plt.axhline(y=miou, color='r', linestyle='--', linewidth=2, label=f'mIoU: {miou:.4f}')
    plt.legend()
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'class_iou.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Class IoU plot saved to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot training curves from JSON logs')
    parser.add_argument('--log-dir', type=str, 
                       default='work_dirs/gta2cs_daformer_rcs_fdthings/logs',
                       help='Path to log directory containing train.json and val.json')
    parser.add_argument('--save-dir', type=str, 
                       default='work_dirs/gta2cs_daformer_rcs_fdthings/plots',
                       help='Directory to save plots. If not specified, plots will be displayed.')
    parser.add_argument('--class-iou', action='store_true',
                       help='Plot per-class IoU bar chart')
    
    args = parser.parse_args()
    print(f"Loading logs from: {args.log_dir}")
    print(f"Saving plots to: {args.save_dir}")
    
    # 绘制训练曲线
    plot_training_curves(args.log_dir, args.save_dir)
    # 绘制类别IoU
    if args.class_iou:
        plot_class_iou(args.log_dir, args.save_dir)
    
    print("\nDone! Plots saved successfully.")

if __name__ == '__main__':
    main()
