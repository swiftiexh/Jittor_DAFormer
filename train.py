# 组装模块（数据、模型、优化器）、控制训练流程（迭代、日志、保存）

import importlib.util
import sys
import random
import numpy as np
import jittor as jt
import os
import glob
import argparse


# 加载配置模块
def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path) # 创建模块描述符
    cfg = importlib.util.module_from_spec(spec) # 创建模块对象
    sys.modules["config"] = cfg # 将模块添加到 sys.modules 中，使其可被导入
    spec.loader.exec_module(cfg) # 执行模块代码，加载配置
    return cfg

# 设置随机种子以确保可复现性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)

# 根据配置列表构建 pipeline
def build_pipeline(pipeline_cfg_list):
    from utils import transform as transforms
    pipeline = []
    for cfg in pipeline_cfg_list:
        cfg = cfg.copy()
        transform_type = cfg.pop('type')
        # 动态获取 transform 类
        transform_cls = getattr(transforms, transform_type)
        pipeline.append(transform_cls(**cfg))
    return pipeline

# 构建训练集
def build_dataset(data_cfg):
    from datasets.gta import GTADataset
    from datasets.cityscapes import CityscapesDataset
    from datasets.uda_dataset import UDADataset

    # 构建源域数据集
    source_cfg = data_cfg['train']['source']
    source_pipeline = build_pipeline(source_cfg['pipeline'])
    source_dataset = GTADataset(
        data_root=source_cfg['data_root'],
        img_dir=source_cfg['img_dir'],
        ann_dir=source_cfg['ann_dir'],
        pipeline=source_pipeline
    )

    # 构建目标域数据集
    target_cfg = data_cfg['train']['target']
    target_pipeline = build_pipeline(target_cfg['pipeline'])
    target_dataset = CityscapesDataset(
        data_root=target_cfg['data_root'],
        img_dir=target_cfg['img_dir'],
        ann_dir=target_cfg['ann_dir'],
        pipeline=target_pipeline
    )

    # 封装为 UDA 数据集
    uda_dataset = UDADataset(
        source=source_dataset,
        target=target_dataset,
        rare_class_sampling=data_cfg['train'].get('rare_class_sampling'),
        batch_size=data_cfg['samples_per_gpu'],
        num_workers=data_cfg['workers_per_gpu'],
        shuffle=True
    )

    return uda_dataset

# 构建验证集
def build_val_dataset(data_cfg):
    from datasets.cityscapes import CityscapesDataset
    
    val_cfg = data_cfg['val']
    val_pipeline = build_pipeline(val_cfg['pipeline'])
    val_dataset = CityscapesDataset(
        data_root=val_cfg['data_root'],
        img_dir=val_cfg['img_dir'],
        ann_dir=val_cfg['ann_dir'],
        pipeline=val_pipeline,
        batch_size=1,  # 验证时 batch_size=1
        num_workers=data_cfg['workers_per_gpu'],
        shuffle=False  # 验证时不需要 shuffle
    )
    return val_dataset

# 构建模型
def build_model(model_cfg):
    from models.segmentor import build_segmentor
    model = build_segmentor(model_cfg)  
    return model

# 构建优化器
def build_optimizer(model, optim_cfg):
    optim_type = optim_cfg.get('type', 'AdamW')
    base_lr = optim_cfg.get('lr', 6e-5)
    weight_decay = optim_cfg.get('weight_decay', 0.01)
    betas = optim_cfg.get('betas', (0.9, 0.999))
    # 参数分组配置
    paramwise_cfg = optim_cfg.get('paramwise_cfg', {})
    custom_keys = paramwise_cfg.get('custom_keys', {})
    # 参数分组：根据名称匹配规则
    params = []
    # 遍历所有参数
    for name, param in model.named_parameters():
        # Jittor 的 Var 没有 requires_grad 属性。Jittor 会根据 stop_grad() 来判断参数是否参与梯度更新
        # 默认配置
        param_group = {
            'params': [param],
            'lr': base_lr,
            'weight_decay': weight_decay
        }
        # 检查是否匹配自定义规则
        for key, config in custom_keys.items():
            if key in name:
                # 应用学习率倍率
                if 'lr_mult' in config:
                    param_group['lr'] = base_lr * config['lr_mult']
                # 应用权重衰减倍率
                if 'decay_mult' in config:
                    param_group['weight_decay'] = weight_decay * config['decay_mult']
                # 找到第一个匹配的规则后跳出
                # 优先级：head > pos_block > norm
                break
        params.append(param_group)
    assert optim_type == 'AdamW', f"Now only AdamW is supported"
    
    # Jittor: 创建 optimizer 时必须传 lr，但会覆盖 param_groups 的设置
    # 需要保存原始的 lr 和 weight_decay，创建后再恢复
    original_lrs = [pg['lr'] for pg in params]
    original_wds = [pg['weight_decay'] for pg in params]
    
    optimizer = jt.optim.AdamW(params, lr=base_lr, betas=betas, weight_decay=weight_decay)
    
    # 恢复每个 param_group 的原始 lr 和 weight_decay
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = original_lrs[i]
        param_group['weight_decay'] = original_wds[i]
    
    return optimizer, original_lrs

# 构建学习率调度器
def build_lr_scheduler(optimizer, lr_config):
    from utils.lr_scheduler import PolyLRWithWarmup
    return PolyLRWithWarmup(
            optimizer=optimizer,
            max_iters=lr_config['max_iters'],
            warmup_iters=lr_config.get('warmup_iters', 1500),
            warmup_ratio=lr_config.get('warmup_ratio', 1e-6),
            power=lr_config.get('power', 1.0),
            min_lr=lr_config.get('min_lr', 0.0)
        )

# 验证函数
def validate(model, val_loader, num_classes=19):
    model.eval()
    # 初始化混淆矩阵：用于计算 IoU
    confusion_matrix = jt.zeros((num_classes, num_classes))
    from tqdm import tqdm 
    with jt.no_grad():
        for batch in tqdm(val_loader, desc='Validating'): 
            img = batch['img']
            gt_seg = batch['gt_semantic_seg']  # [B, 1, H, W]
            # 推理
            pred = model.encode_decode(img)  # [B, num_classes, H, W]
            pred = pred.argmax(dim=1)[0]  # [B, H, W]
            # 展平 
            gt_seg = gt_seg.squeeze(1).reshape(-1)  # [B*H*W]
            pred = pred.reshape(-1)  # [B*H*W]
            # 忽略 ignore_index=255 的像素
            valid_mask = (gt_seg != 255)
            gt_seg = gt_seg[valid_mask]
            pred = pred[valid_mask]
            # 更新混淆矩阵
            # Jittor: 将 tensor 转为 numpy 进行索引操作
            gt_seg_np = gt_seg.numpy()
            pred_np = pred.numpy()
            for t, p in zip(gt_seg_np, pred_np):
                confusion_matrix[int(t), int(p)] += 1
    # 计算 IoU
    iou_per_class = []
    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        
        iou = tp / (tp + fp + fn + 1e-10)
        iou_per_class.append(iou.item())
    mean_iou = sum(iou_per_class) / num_classes
    model.train()
    return {
        'mIoU': mean_iou,
        'IoU_per_class': iou_per_class
    }

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='DAFormer Training')
    parser.add_argument('--config', type=str, default='configs/gta2cs_daformer.py',
                        help='configuration file path (default: configs/gta2cs_daformer.py)')
    parser.add_argument('--resume', type=str, default=None,
                        help='checkpoint path to resume from (default: None)')
    args = parser.parse_args()
    
    # Cityscapes 19类别名称
    CITYSCAPES_CLASSES = [
        'road', 'sidewalk', 'building', 'wall', 'fence',
        'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
        'sky', 'person', 'rider', 'car', 'truck',
        'bus', 'train', 'motorcycle', 'bicycle'
    ]
    # 1. 设置 Jittor 使用 GPU（如果可用）
    if jt.has_cuda:
        jt.flags.use_cuda = 1
        print("Using CUDA (GPU) for training")
    else:
        print("CUDA not available, using CPU")
    # 2. 加载配置
    cfg = load_config(args.config)
    # 设置随机种子
    set_seed(cfg.seed)
    # Jittor 不需要 cudnn benchmark 设置
    # 3. 构建数据集（Dataloader）
    # Jittor 不需要显式构建 Dataloader
    print("Building dataset...")
    train_loader = build_dataset(cfg.data)
    print("Building validation dataset...")
    val_loader = build_val_dataset(cfg.data)
    # 4. 构建模型
    print("Building model...")
    model = build_model(cfg.model)
    # 5. 构建优化器和学习率调度器
    optimizer, original_lrs = build_optimizer(model, cfg.optim)    
    lr_scheduler = build_lr_scheduler(optimizer, cfg.lr_schedule)
    # 6. 构建 Trainer（封装 UDA 逻辑）
    from trainer.train_daformer import DAFormerTrainer
    trainer = DAFormerTrainer(
        model=model,
        uda_cfg=cfg.uda, # UDA 相关配置，如损失权重、伪标签更新频率等
        optimizer=optimizer,
        lr_schedule=lr_scheduler, # 学习率调度配置，如 warmup、step decay 等
        runner_cfg=cfg.runner # 训练流程配置
    )
    # 6. 训练循环
    # 创建 checkpoint 和 log 目录
    checkpoint_dir = f'work_dirs/{cfg.name}'
    log_dir = os.path.join(checkpoint_dir, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    # 初始化日志记录器
    from utils.logger import JSONLogger
    logger = JSONLogger(log_dir, cfg.name)
    # 如果指定了resume，加载checkpoint
    start_iteration = 0
    if args.resume:
        from utils.checkpoint import load_checkpoint
        start_iteration = load_checkpoint(
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema_model=trainer.ema_model
        )
        # 清理checkpoint之后的日志，防止重复
        logger.truncate_logs(start_iteration)
        print(f"Resume training from iteration {start_iteration}")
    else:
        # 记录配置信息(仅在首次训练时)
        logger.write_config({
            'max_iters': cfg.runner['max_iters'],
            'batch_size': cfg.data['samples_per_gpu'],
            'base_lr': cfg.optim['lr'],
            'weight_decay': cfg.optim.get('weight_decay', 0.01)
        })
    # 创建循环数据加载器
    from itertools import cycle
    train_loader_iter = cycle(train_loader)
    for iteration in range(start_iteration + 1, cfg.runner['max_iters'] + 1):
        # 获取一个 batch
        batch = next(train_loader_iter)
        # Jittor 无需显式地把数据移动到 GPU，数据会自动在 GPU 上处理
        # 执行一步训练
        log_vars = trainer.train_step(batch, iteration)
        # 打印日志并记录到文件
        if iteration % cfg.log['interval'] == 0:
            log_str = f"Iter [{iteration}/{cfg.runner['max_iters']}]"
            for key, val in log_vars.items():
                # 提取 tensor 的标量值
                if isinstance(val, jt.Var):
                    val_scalar = val.item()
                else:
                    val_scalar = val
                # lr 和 grad_norm 使用科学计数法，其他使用4位小数
                if key in ['lr', 'grad_norm']:
                    log_str += f" {key}: {val_scalar:.4e}"
                else:
                    log_str += f" {key}: {val_scalar:.4f}"
            print(log_str)
            # 使用logger记录
            logger.log_train(iteration, log_vars)
        # 定期深度清理显存
        if iteration % 50 == 0:
            jt.sync_all()  # 同步所有操作
            jt.clean_graph()  # 深度清理计算图
            jt.gc()  # 垃圾回收
        # 验证
        if iteration % cfg.evaluation['interval'] == 0 and iteration > 0:
            print(f"\nEvaluating at iter {iteration}...")
            eval_results = validate(model, val_loader, num_classes=19)
            print(f"mIoU: {eval_results['mIoU']:.4f}")
            for i, iou in enumerate(eval_results['IoU_per_class']):
                print(f"{CITYSCAPES_CLASSES[i]:15s}: {iou:.4f}")
            # 使用logger记录验证结果
            logger.log_val(iteration, eval_results)
            # 验证后立即清理显存
            jt.clean_graph()
            jt.gc()
            print()  
        # 保存 checkpoint
        if iteration % cfg.checkpoint_config['interval'] == 0 and iteration > 0:
            print(f"\nSaving checkpoint at iter {iteration}...")
            from utils.checkpoint import save_checkpoint
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                ema_model=trainer.ema_model,
                iteration=iteration,
                checkpoint_dir=checkpoint_dir,
                max_keep_ckpts=cfg.checkpoint_config['max_keep_ckpts']
            )
            # checkpoint保存后立即清理显存
            jt.clean_graph()
            jt.gc()
            print() 

if __name__ == '__main__':
    main()