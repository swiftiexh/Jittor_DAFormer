# 实验配置

# 数据集设置：使用 GTA 数据集作为源域，Cityscapes 作为目标域
# 三类数据处理流水线：GTA 训练流水线、Cityscapes 训练流水线、测试流水线
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True) # 图像归一化参数
crop_size = (384, 384) # 裁剪尺寸
gta_train_pipeline = [
    dict(type='LoadImageFromFile'), # 从文件加载图像
    dict(type='LoadAnnotations'), # 加载标注
    dict(type='Resize', img_scale=(1280, 720)), # 调整图像大小
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75), # 随机裁剪，类别最大比例限制
    dict(type='RandomFlip', prob=0.5), # 随机水平翻转
    # dict(type='PhotoMetricDistortion'), # 光度畸变增强（可选）
    dict(type='Normalize', **img_norm_cfg), # 图像归一化
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),  # 填充图像和标注，pad_val为图像填充值，seg_pad_val为分割标注填充值
    dict(type='DefaultFormatBundle'), # 默认格式打包，包括图像和标注的格式转换
    dict(type='Collect', keys=['img', 'gt_semantic_seg']), # 收集图像和语义分割标注
]
cityscapes_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 512)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'), 
    dict(type='Resize', img_scale=(1024, 512), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img','gt_semantic_seg']),
]
# 数据集定义
data = dict(
    samples_per_gpu=1, # 每个 GPU 上的 batch size
    workers_per_gpu=4, # 每个 GPU 对应的数据加载线程/进程数
    train = dict(
        type='UDADataset',
        source = dict(
            type='GTADataset',
            data_root='data/gta/', # GTA 数据集根目录
            img_dir='images', # 图像目录
            ann_dir='labels', # 标注目录
            pipeline=gta_train_pipeline # GTA 训练流水线
        ),
        target = dict(
            type='CityscapesDataset',
            data_root='data/cityscapes/', # Cityscapes 数据集根目录
            img_dir='leftImg8bit/train', # 训练图像目录
            ann_dir='gtFine/train', # 训练标注目录
            pipeline=cityscapes_train_pipeline # Cityscapes 训练流水线
        ),
        # RCS 流程：确定候选类别集合 -> 计算类别频率 -> 计算采样概率 -> 执行采样
        rare_class_sampling = dict(
            min_pixels = 3000, class_temp = 0.01, min_crop_ratio = 0.5) # 稀有类采样配置：最小像素数、类别温度、最小裁剪比例
    ),
    val=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline
    ),
    test=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline
    )
)
# --------------------------------------------------------------------------------------------------------------
# 模型设置
norm_cfg = dict(type='BN', requires_grad=True) # 归一化层配置
model = dict(
    type='EncoderDecoder', 
    pretrained='pretrained/mit_b5.pth', # MiT-B5 预训练权重路径

    # Backbone: MiT-B5 
    backbone=dict(
        type='mit_b5',  # 对应 models/backbones/mit_b5.py
        style='pytorch'  # PyTorch 风格
    ),

    # Decode Head: DAFormer with Sep-ASPP
    decode_head=dict(
        type='DAFormerHead',
        in_channels=[64, 128, 320, 512],  # MiT-B5 四个阶段的输出通道数 
        in_index=[0, 1, 2, 3],  # 使用所有四个阶段的特征 
        channels=256,  # 融合后的通道数 
        dropout_ratio=0.1,  # Dropout 概率，在最终分割前用于防止过拟合 
        num_classes=19,  # Cityscapes 19 类 
        norm_cfg=norm_cfg,  # 归一化配置 
        align_corners=False,  # 双线性插值不对齐角点，而是对齐块 
        # Decoder 参数
        decoder_params=dict(
            embed_dims=256,  # 嵌入维度
            # 中间层特征的嵌入配置（MLP，无激活和归一化）
            # Stages 1-3 的嵌入均采用相同配置
            embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            # 最高层特征的嵌入配置（MLP，无激活和归一化）
            # Stage 4 的嵌入采用相同配置
            embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            # 融合配置：深度可分离 ASPP
            fusion_cfg=dict(
                type='aspp',
                sep=True,  # 使用深度可分离卷积
                dilations=(1, 6, 12, 18),  # 空洞率
                pool=False,  # 不使用全局平均池化分支
                act_cfg=dict(type='ReLU'), # 激活函数
                norm_cfg=norm_cfg
            )
        ),
        # 损失函数配置
        loss_decode=dict(
            type='CrossEntropyLoss',  # 交叉熵损失
            use_sigmoid=False,  # 使用 softmax（多分类）
            loss_weight=1.0
        )
    ),

    # 训练和测试配置（来自 daformer_conv1_mitb5.py）
    train_cfg=dict(),  # 训练时的额外配置（这里为空）
    test_cfg=dict(mode='whole')  # 测试时使用整图推理（vs. sliding window）
)
# --------------------------------------------------------------------------------------------------------------
# UDA 训练设置
uda = dict(
    type='DACS',  # UDA 方法类型：DACS（基于类别混合的自适应域自训练）

    # EMA Teacher 配置
    alpha=0.999,  # EMA 动量

    # 伪标签配置
    pseudo_threshold=0.968,  # 伪标签置信度阈值（来自 dacs.py baseline）
    pseudo_weight_ignore_top=15,  # 忽略图像顶部 15 像素（来自顶层配置）
    pseudo_weight_ignore_bottom=120,  # 忽略图像底部 120 像素（防止伪影）

    # ImageNet 特征距离损失
    imnet_feature_dist_lambda=0.005,  # 特征距离损失权重
    imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],  # Thing classes
    # 对应：traffic_light, traffic_sign, person, rider, car, truck, bus, train, motorcycle, bicycle
    imnet_feature_dist_scale_min_ratio=0.75,  # 特征图缩放的最小比例

    # ClassMix 配置
    mix='class',  # 使用类别级混合（ClassMix）
    blur=True,  # 对混合图像应用高斯模糊
    color_jitter_strength=0.2,  # 颜色抖动强度
    color_jitter_probability=0.2,  # 颜色抖动概率
)
# --------------------------------------------------------------------------------------------------------------
# 优化器设置
optim = dict(
    type='AdamW',  # AdamW 优化器
    lr=6e-5,  # 基础学习率
    betas=(0.9, 0.999),  # Adam 的 beta 参数
    weight_decay=0.01,  # 权重衰减

    # 参数分组配置
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),  # Decode head 使用 10 倍学习率
            pos_block=dict(decay_mult=0.0),  # 位置编码块无权重衰减
            norm=dict(decay_mult=0.0)  # 归一化层（BN/LN）无权重衰减
        )
    )
)
# --------------------------------------------------------------------------------------------------------------
# 学习率调度设置
lr_schedule = dict(
    type='poly',  # Poly 学习率衰减
    warmup='linear',  # 线性 warmup
    warmup_iters=1500,  # Warmup 迭代数
    warmup_ratio=1e-6,  # Warmup 起始学习率 = lr * warmup_ratio
    power=1.0,  # Poly 的幂次（1.0 表示线性衰减）
    min_lr=0.0,  # 最小学习率
    by_epoch=False,  # 按迭代数调度
    max_iters=40000  # 最大迭代数
)
# --------------------------------------------------------------------------------------------------------------
# 训练流程设置
runner = dict(
    type='IterBasedRunner',  # 基于迭代的训练
    max_iters=40000  # 总迭代数：40k
)
# 日志配置
log = dict(
    interval=50  # 每 50 iter 打印一次日志
)
# Checkpoint 配置
checkpoint_config = dict(
    interval=500,  # 每 500 iter 保存一次 checkpoint，支持断点续训
    max_keep_ckpts=4  # 最多保留 4 个 checkpoint
)
# 验证配置
evaluation = dict(
    interval=10000,  # 每 10000 iter 验证一次
    metric='mIoU'  # 评估指标：mean IoU
)
# 随机种子
seed = 0
# 实验名称
name = 'gta2cs_daformer_rcs_fdthings'