# 实现多项式学习率调度器，包含预热阶段

class PolyLRWithWarmup:
    def __init__(self, optimizer, max_iters, warmup_iters=1500, 
            warmup_ratio=1e-6, power=1.0, min_lr=0.0):
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.power = power
        self.min_lr = min_lr

        # 保存每个参数组的初始学习率
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        # 当前迭代计数
        self.last_iter = 0
    
    def step(self):
        self.last_iter += 1
        # 为每个参数组计算新的学习率
        for i, param_group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            new_lr = self._compute_lr(base_lr, self.last_iter)
            param_group['lr'] = new_lr

    def _compute_lr(self, base_lr, current_iter):
        if current_iter < self.warmup_iters:
            # Warmup 阶段: 线性增长
            k = (1 - self.warmup_ratio) * current_iter / self.warmup_iters + self.warmup_ratio
            lr = base_lr * k
        else:
            # Poly 衰减阶段
            progress = (current_iter - self.warmup_iters) / (self.max_iters - self.warmup_iters)
            lr = (base_lr - self.min_lr) * pow(1 - progress, self.power) + self.min_lr
            # 确保学习率不低于最小值
            lr = max(lr, self.min_lr)
        return lr
    
    def get_last_lr(self):  
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        return {
            'last_iter': self.last_iter,
            'base_lrs': self.base_lrs,
            'max_iters': self.max_iters,
            'warmup_iters': self.warmup_iters,
            'warmup_ratio': self.warmup_ratio,
            'power': self.power,
            'min_lr': self.min_lr
        }

    def load_state_dict(self, state_dict):
        self.last_iter = state_dict['last_iter']
        self.base_lrs = state_dict['base_lrs']
        self.max_iters = state_dict['max_iters']
        self.warmup_iters = state_dict['warmup_iters']
        self.warmup_ratio = state_dict['warmup_ratio']
        self.power = state_dict['power']
        self.min_lr = state_dict['min_lr']