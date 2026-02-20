# 

from copy import deepcopy
import jittor as jt

class DAFormerTrainer:
    def __init__(self, model, uda_cfg, optimizer, lr_schedule, runner_cfg):
        self.model = model  # 学生模型
        self.uda_cfg = uda_cfg
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.max_iters = runner_cfg['max_iters']

        # UDA 超参数
        self.alpha = uda_cfg.get('alpha', 0.99)  # EMA 更新权重
        self.pseudo_threshold = uda_cfg.get('pseudo_threshold', 0.968) # 伪标签置信度阈值
        # Therefore, we ignore the top 15 and bottom 120 pixels of the pseudo-label
        self.pseudo_weight_ignore_top = uda_cfg.get('pseudo_weight_ignore_top', 0) # 伪标签权重忽略图像顶部像素行数
        self.pseudo_weight_ignore_bottom = uda_cfg.get('pseudo_weight_ignore_bottom', 0) # 伪标签权重忽略图像底部像素行数

        # 特征距离
        self.imnet_feature_dist_lambda = uda_cfg.get('imnet_feature_dist_lambda', 0) # ImageNet 特征距离损失权重
        self.imnet_feature_dist_classes = uda_cfg.get('imnet_feature_dist_classes', None) # 计算特征距离的类别列表（如 [0, 1, 2]），如果为 None 则对所有类别计算
        self.imnet_feature_dist_scale_min_ratio = uda_cfg.get('imnet_feature_dist_scale_min_ratio', None) # 计算特征距离的最小尺度比例（如 0.5），如果为 None 则不进行尺度过滤
        self.enable_fdist = self.imnet_feature_dist_lambda > 0 # 是否启用特征距离

        # 混合增强
        self.mix = uda_cfg.get('mix', 'class') # 混合增强类型：'class'（类别混合）
        self.blur = uda_cfg.get('blur', True) # 是否对混合图像进行模糊处理以减少边界伪影
        self.color_jitter_strength = uda_cfg.get('color_jitter_strength', 0.2) # 颜色抖动强度
        self.color_jitter_probability = uda_cfg.get('color_jitter_probability', 0.2) # 颜色抖动概率

        # 初始化 EMA 教师（使用 models/ema.py）
        from models.ema import EMAModel
        self.ema_model = EMAModel(model, alpha=self.alpha)

        # 初始化特征距离模型
        if self.enable_fdist:
            self.imnet_model = deepcopy(model)
            for param in self.imnet_model.parameters():
                param.requires_grad = False
            self.imnet_model.eval()
        else:
            self.imnet_model = None

    # 执行一步 UDA 训练
    def train_step(self, batch, iteration):
        self.model.train()
        # Jittor: optimizer.step() 会自动 zero_grad()
        log_vars = {}

        total_loss = 0.0
        # 1. 源域监督训练
        src_losses, src_feat = self._train_on_source(
            batch['img'],
            batch['gt_semantic_seg'],
        )
        src_loss = src_losses['loss']
        
        # 2. 特征距离损失（如果启用）
        if self.enable_fdist and src_feat is not None:
            fdist_loss = self._calc_feat_dist(
                batch['img'],
                batch['gt_semantic_seg'],
                src_feat
            )
            # Jittor: 累积损失后一起 backward，避免 retain_graph 问题
            src_loss = src_loss + fdist_loss
            log_vars['loss_imnet_feat_dist'] = fdist_loss.item()
            total_loss += fdist_loss.item()
            del src_feat, fdist_loss
        
        # 反向传播源域损失（可能包含特征距离损失）
        # Jittor: 使用 optimizer.backward(loss) 而不是 loss.backward()
        self.optimizer.backward(src_loss)
        total_loss += src_loss.item()
        log_vars.update({f'src_{k}': v.item() for k, v in src_losses.items()})
        del src_loss, src_losses

        # 3. 伪标签生成
        with jt.no_grad():
            pseudo_label, pseudo_weight = self._generate_pseudo_label(batch['target_img'])
        log_vars['pseudo_ratio'] = (pseudo_weight > 0).float32().mean().item()

        # 4. 混合增强
        mixed_img, mixed_lbl, mixed_weight = self._apply_class_mix(
            batch['img'],
            batch['gt_semantic_seg'],
            batch['target_img'],
            pseudo_label,
            pseudo_weight,
        ) # each: [B,3,H,W], [B,H,W], [B,H,W]
        del pseudo_label, pseudo_weight

        # 5. 混合图像训练
        mix_losses = self._train_on_mixed(
            mixed_img,
            mixed_lbl,
            mixed_weight
        )
        mix_loss = mix_losses['loss']
        # Jittor: 使用 optimizer.backward(loss)
        self.optimizer.backward(mix_loss)
        total_loss += mix_loss.item()
        log_vars.update({f'mix_{k}': v.item() for k, v in mix_losses.items()})
        del mix_loss, mix_losses, mixed_img, mixed_lbl, mixed_weight

        # 6. 梯度裁剪，防止梯度爆炸
        # Jittor: 梯度存储在 optimizer.param_groups[i]['grads'] 中
        total_norm = 0.0
        for param_group in self.optimizer.param_groups:
            if 'grads' in param_group:
                for grad in param_group['grads']:
                    if grad is not None:
                        param_norm = grad.sqr().sum()
                        total_norm += param_norm
        total_norm = (total_norm ** 0.5).item()
        # 如果梯度范数超过阈值，进行裁剪
        max_grad_norm = 1.0  # 梯度裁剪阈值
        if total_norm > max_grad_norm:
            clip_coef = max_grad_norm / (total_norm + 1e-6)
            for param_group in self.optimizer.param_groups:
                if 'grads' in param_group:
                    for i, grad in enumerate(param_group['grads']):
                        if grad is not None:
                            param_group['grads'][i] = grad * clip_coef
        log_vars['grad_norm'] = total_norm
        
        # 7. 优化器步进
        self.optimizer.step()

        # 8. 更新 EMA 教师
        self.ema_model.update(iteration, self.model)

        # 9. 学习率调度
        self.lr_schedule.step()
        log_vars['lr'] = self.lr_schedule.get_last_lr()[0]

        log_vars['total_loss'] = total_loss

        # 最后清理计算图和缓存
        jt.clean_graph() 
        jt.gc()

        return log_vars

    # 在源域上计算监督损失，并提取特征
    def _train_on_source(self, img, gt_seg):
        # 调用 models/segmentor.py 的 forward_train
        outputs = self.model.forward_train(
            img=img, # [B,3,H,W]
            gt_semantic_seg=gt_seg, # [B,1,H,W] 
            return_feat=self.enable_fdist 
        )
        # 提取特征，用于特征距离计算
        features = outputs.get('features', None)
        loss_seg = outputs['loss_seg']
        acc_seg = outputs['acc_seg']
        losses = {'loss':loss_seg,'acc':acc_seg}
        return losses, features
    
    # 计算 ImageNet 特征距离损失
    def _calc_feat_dist(self, img, gt_seg, student_feat):
        with jt.no_grad():
            imnet_feat = self.imnet_model.backbone(img) 
        # 调用 utils/losses.py 中的计算特征距离的函数
        from utils.losses import masked_feature_distance
        feat_loss = masked_feature_distance(
            student_feat[-1],  #  [B, 512, H/32, W/32] （the bottleneck features）
            imnet_feat[-1], #  [B, 512, H/32, W/32] （the bottleneck features）
            gt_seg, # [B, 1, H, W]
            mask_classes=self.imnet_feature_dist_classes, #  Thing-Class
            scale_min_ratio=self.imnet_feature_dist_scale_min_ratio # 最小尺度比例
        )
        return self.imnet_feature_dist_lambda * feat_loss
    
    # 生成伪标签
    def _generate_pseudo_label(self, target_img):
        from utils.pseudo_label import generate_pseudo_label
        # 使用 EMA 教师模型生成伪标签
        with jt.no_grad():
            ema_logits = self.ema_model.model.encode_decode(target_img)
        # 调用工具函数
        pseudo_label, pseudo_weight = generate_pseudo_label(
            ema_logits,
            threshold=self.pseudo_threshold,
            ignore_top=self.pseudo_weight_ignore_top,
            ignore_bottom=self.pseudo_weight_ignore_bottom
        )
        del ema_logits  # 释放logits显存
        # 返回伪标签和权重 [B, H, W]
        return pseudo_label, pseudo_weight
    
    # 应用类别混合增强
    def _apply_class_mix(self, src_img, src_seg, tgt_img, pseudo_label,
                        pseudo_weight):
        from utils.mix import apply_class_mix
        # 调用工具函数
        mixed_img, mixed_lbl, mixed_weight = apply_class_mix(
            src_img=src_img, # [B,3,H,W]
            src_seg=src_seg, # [B,1,H,W]
            tgt_img=tgt_img, # [B,3,H,W]
            pseudo_label=pseudo_label, # [B,H,W]
            pseudo_weight=pseudo_weight, # [B,H,W]
            mix_type=self.mix,
            blur=self.blur,
            color_jitter_s=self.color_jitter_strength,
            color_jitter_p=self.color_jitter_probability
        ) # each: [B,3,H,W], [B,H,W], [B,H,W]
        # 返回混合图像、标签和权重 
        return mixed_img, mixed_lbl, mixed_weight 
    
    # 在混合图像上计算损失
    def _train_on_mixed(self, mixed_img, mixed_lbl, mixed_weight):
        mixed_lbl = mixed_lbl.unsqueeze(1)  # [B,H,W] -> [B,1,H,W]
        outputs = self.model.forward_train(
            img=mixed_img,
            gt_semantic_seg=mixed_lbl,
            seg_weight=mixed_weight,
            return_feat=False
        )
        loss_seg = outputs['loss_seg']
        acc_seg = outputs['acc_seg']
        losses = {'loss':loss_seg,'acc':acc_seg}
        return losses