# 与 checkpoint 相关的辅助函数

import os
import random
import numpy as np
import jittor as jt
import glob
import time

# 保存 checkpoint
def save_checkpoint(model, optimizer, lr_scheduler, ema_model, iteration, checkpoint_dir, max_keep_ckpts=1):
    checkpoint_path = os.path.join(checkpoint_dir, f'iter_{iteration}.pth')
    # 手动保存optimizer的param_groups，因为 Jittor 可能不完整保存
    param_groups_state = []
    for pg in optimizer.param_groups:
        pg_state = {
            'lr': pg['lr'],
            'weight_decay': pg['weight_decay'],
            'betas': pg.get('betas', (0.9, 0.999)),
            'eps': pg.get('eps', 1e-8)
        }
        param_groups_state.append(pg_state)   
    # 保存随机状态
    random_states = {
        'random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
    }
    checkpoint = {
        'iteration': iteration,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'optimizer_param_groups': param_groups_state,  # 额外保存param_groups
        'lr_scheduler': lr_scheduler.state_dict(),
        'ema_model': ema_model.state_dict(),
        **random_states  # 保存所有随机状态
    }
    jt.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    # 保留最新的 max_keep_ckpts 个
    if max_keep_ckpts > 0:
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'iter_*.pth'))
        # 按文件名中的迭代数排序（文件名形如 iter_<num>.pth），避免字符串排序错误
        def _ckpt_iter(path):
            base = os.path.basename(path)
            try:
                num_str = base.replace('iter_', '').replace('.pth', '')
                return int(num_str)
            except Exception:
                return -1
        checkpoint_files = sorted(checkpoint_files, key=_ckpt_iter)
        if len(checkpoint_files) > max_keep_ckpts:
            to_remove = checkpoint_files[:-max_keep_ckpts]
            for old_ckpt in to_remove:
                try:
                    os.remove(old_ckpt)
                    print(f"Removed old checkpoint: {old_ckpt}")
                except Exception as e:
                    print(f"Warning: failed to remove old checkpoint {old_ckpt}: {e}")

# 加载 checkpoint
def load_checkpoint(checkpoint_path, model, optimizer, lr_scheduler, ema_model):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = jt.load(checkpoint_path)
    # 1. 加载模型状态
    model.load_state_dict(checkpoint['model'])
    # 2. 加载优化器状态
    optimizer.load_state_dict(checkpoint['optimizer'])
    # 3. 恢复optimizer的param_groups
    if 'optimizer_param_groups' in checkpoint:
        param_groups_state = checkpoint['optimizer_param_groups']
        if len(param_groups_state) == len(optimizer.param_groups):
            for pg, pg_state in zip(optimizer.param_groups, param_groups_state):
                pg['lr'] = pg_state['lr']
                pg['weight_decay'] = pg_state['weight_decay']
                if 'betas' in pg_state:
                    pg['betas'] = pg_state['betas']
                if 'eps' in pg_state:
                    pg['eps'] = pg_state['eps']
        else:
            print(f"Warning: param_groups count mismatch ({len(param_groups_state)} vs {len(optimizer.param_groups)})")
    else:
        print(f"Warning: No optimizer_param_groups in checkpoint (old format)")
    # 4. 恢复随机状态
    restored_random_states = []
    if 'random_state' in checkpoint:
        random.setstate(checkpoint['random_state'])
        restored_random_states.append('Python random')
    if 'numpy_random_state' in checkpoint:
        np.random.set_state(checkpoint['numpy_random_state'])
        restored_random_states.append('NumPy random')
    # 5. 加载其他状态
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    ema_model.load_state_dict(checkpoint['ema_model'])
    start_iteration = checkpoint['iteration']
    print(f"Successfully resumed from iteration {start_iteration}")
    return start_iteration

# 查找最新的 checkpoint
def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'iter_*.pth'))
    if not checkpoint_files:
        return None
    # 按文件名中的迭代数排序
    def _ckpt_iter(path):
        base = os.path.basename(path)
        try:
            num_str = base.replace('iter_', '').replace('.pth', '')
            return int(num_str)
        except Exception:
            return -1
    checkpoint_files = sorted(checkpoint_files, key=_ckpt_iter)
    latest_checkpoint = checkpoint_files[-1] if checkpoint_files else None
    return latest_checkpoint