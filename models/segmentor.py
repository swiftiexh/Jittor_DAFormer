# 学生模型

import jittor as jt
from jittor import nn

class Segmentor(nn.Module):
    def __init__(self, backbone, decode_head, num_classes=19, align_corners=False):
        super().__init__()
        self.backbone = backbone
        self.decode_head = decode_head
        self.num_classes = num_classes
        self.align_corners = align_corners
    
    # 训练前向传播，返回损失字典
    def forward_train(self, 
                     img, 
                     gt_semantic_seg, 
                     seg_weight=None, 
                     return_feat=False):
        losses = {}
        # 提取特征
        features = self.backbone(img)
        # 解码头前向传播
        seg_logits = self.decode_head(features)
        # 计算损失
        loss_dict = self._compute_loss(seg_logits, gt_semantic_seg, seg_weight)
        losses.update(loss_dict)
        if return_feat:
            losses['features'] = features
        return losses
    
    # 计算损失
    def _compute_loss(self, seg_logits, gt_semantic_seg, seg_weight=None):
        loss_dict = {}
        # 上采样到 gt 大小
        seg_logits = jt.nn.interpolate(
            seg_logits, # Jittor 的 interpolate 函数没有 input 参数
            size=gt_semantic_seg.shape[2:],  # (H, W)
            mode='bilinear',
            align_corners=self.align_corners
        )  
        # Jittor 没有 .long() 方法，使用 .int64() 替代
        gt_semantic_seg = gt_semantic_seg.squeeze(1).int64() # (B, H, W)
        # 计算交叉熵损失
        if seg_weight is not None:
            loss_seg = jt.nn.cross_entropy_loss(
                seg_logits,  # (B, C, H, W)
                gt_semantic_seg,  # (B, H, W)
                ignore_index=255,
                reduction='none'  # 不进行自动归约
            )   # (B, H, W)
            # 应用权重加权平均
            loss_seg = (loss_seg * seg_weight).sum() / (seg_weight.sum() + 1e-8) 
        else:
            loss_seg = jt.nn.cross_entropy_loss(
                seg_logits,  # (B, C, H, W)
                gt_semantic_seg,  # (B, H, W)
                ignore_index=255,
                reduction='mean'  # 平均损失
            )
        loss_dict['loss_seg'] = loss_seg
        # 计算精度
        with jt.no_grad():
            seg_pred = seg_logits.argmax(dim=1)[0]  # (B, H, W)
            valid_mask = (gt_semantic_seg != 255)
            # 仅在 valid 区域计算精度
            valid_count = valid_mask.sum() # 类型是 jt.Var，不能直接使用 .item() 获取数值，而应该使用 .data[0] 来访问数值
            if int(valid_count.data[0]) > 0:
                correct = (seg_pred == gt_semantic_seg).logical_and(valid_mask)
                acc_seg = correct.float32().mean()
            else:
                acc_seg = jt.array(0.0)
            loss_dict['acc_seg'] = acc_seg
        return loss_dict
    
    # 推理前向传播
    def encode_decode(self, img):
        # 提取多尺度特征
        features = self.backbone(img)
        # 解码头前向
        seg_logits = self.decode_head(features)  # (B, num_classes, H/4, W/4)
        # 上采样到输入尺寸
        seg_logits = jt.nn.interpolate(
            seg_logits,
            size=img.shape[2:],  # (H, W)
            mode='bilinear',
            align_corners=self.align_corners
        )
        return seg_logits

# 根据配置构建分割模型
def build_segmentor(model_cfg):
    from models.backbones.mit_b5 import mit_b5
    from models.decode_heads.daformer_head import DAFormerHead

    # 1. 构建 backbone
    pretrained = model_cfg.get('pretrained', None)
    backbone = mit_b5(pretrained=pretrained)

    # 2. 构建 decode_head
    head_cfg = model_cfg['decode_head']
    decode_head = DAFormerHead(
        in_channels=head_cfg['in_channels'],
        in_index=head_cfg['in_index'],
        channels=head_cfg['channels'],
        dropout_ratio=head_cfg['dropout_ratio'],
        num_classes=head_cfg['num_classes'],
        norm_cfg=head_cfg['norm_cfg'],
        align_corners=head_cfg['align_corners'],
        decoder_params=head_cfg['decoder_params']
    )

    # 3. 组合为完整模型
    model = Segmentor(
        backbone=backbone,
        decode_head=decode_head,
        num_classes=head_cfg['num_classes'],
        align_corners=head_cfg['align_corners']
    )

    return model