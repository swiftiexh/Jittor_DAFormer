# 解码器：多尺度特征融合 + 深度可分离 ASPP

import jittor as jt
from jittor import nn

# MLP 嵌入层：将输入特征投影到指定维度
class MLP(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def execute(self, x):
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.proj(x)  # (B, H*W, embed_dim)
        return x

# 标准卷积模块：卷积 + 归一化 + 激活
class ConvModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 norm_cfg=None,
                 act_cfg=None):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        # 归一化层
        if norm_cfg is not None:
            if norm_cfg['type'] == 'BN':
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm_cfg['type'] == 'SyncBN':
                self.norm = nn.SyncBatchNorm(out_channels)
            elif norm_cfg['type'] == 'LN':
                self.norm = nn.GroupNorm(1, out_channels) 
            else:
                self.norm = None
        else:
            self.norm = None
        # 激活函数
        if act_cfg is not None:
            if act_cfg['type'] == 'ReLU':
                self.activate = nn.ReLU() # Jittor 的 ReLU 不需要 inplace 参数
            elif act_cfg['type'] == 'GELU':
                self.activate = nn.GELU()
            else:
                self.activate = None
        else:
            self.activate = None
    
    def execute(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activate is not None:
            x = self.activate(x)
        return x
    
# 深度可分离卷积模块
class DepthwiseSeparableConvModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 norm_cfg=None,
                 act_cfg=None):
        super().__init__()
        # Depthwise: 每个输入通道单独卷积
        self.depthwise = ConvModule(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=False,
            norm_cfg=norm_cfg, act_cfg=act_cfg)
        # Pointwise: 1x1 卷积融合通道
        self.pointwise = ConvModule(
            in_channels, out_channels, kernel_size=1,
            bias=False, norm_cfg=norm_cfg, act_cfg=act_cfg)
        
    def execute(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# 深度可分离 ASPP 模块的多个分支
# 由于 ASPP 的分支数量由 dilations 决定，因此实现为 nn.ModuleList
class DepthwiseSeparableASPPModule(nn.ModuleList):
    def __init__(self, dilations, in_channels, channels, norm_cfg, act_cfg):
        super().__init__()
        self.dilations = dilations

        for dilation in dilations:
            if dilation == 1:
                # dilation=1 时使用标准 1x1 卷积
                self.append(
                    ConvModule(
                        in_channels, channels, kernel_size=1,
                        norm_cfg=norm_cfg, act_cfg=act_cfg)) 
            else:
                # dilation>1 时使用深度可分离卷积
                self.append(
                    DepthwiseSeparableConvModule(
                        in_channels, channels, kernel_size=3,
                        dilation=dilation, padding=dilation,
                        norm_cfg=norm_cfg, act_cfg=act_cfg))
    
    def execute(self, x):
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))
        return aspp_outs
    
# 深度可分离 ASPP 模块
class ASPPWrapper(nn.Module):
    def __init__(self,
                 in_channels,
                 channels,
                 sep,
                 dilations,
                 pool,
                 norm_cfg,
                 act_cfg,
                 align_corners):
        super().__init__()
        self.dilations = dilations
        self.align_corners = align_corners

        assert sep is True, "Only depthwise separable ASPP is implemented in DAFormer."
        assert pool is False, "Global pooling branch is not implemented in DAFormer."

        # 深度可分离 ASPP 模块 
        self.aspp_modules =  DepthwiseSeparableASPPModule( 
            dilations=dilations,
            in_channels=in_channels,
            channels=channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        
        # Bottleneck: 融合所有 ASPP 分支
        self.bottleneck = ConvModule(
            len(dilations) * channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        
    def execute(self, x):
        aspp_outs = []
        # ASPP 分支
        aspp_outs.extend(self.aspp_modules(x))
        # 拼接并融合
        aspp_outs = jt.concat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        return output


# 动态构建层的工厂函数
def build_layer(in_channels, out_channels, type, **kwargs):
    if type == 'id':
        return nn.Identity()
    elif type == 'mlp':
        return MLP(input_dim=in_channels, embed_dim=out_channels)
    elif type == 'aspp':
        return ASPPWrapper(in_channels=in_channels,channels=out_channels,**kwargs)
    
# 双线性插值上采样函数
def resize(input, size=None, scale_factor=None, mode='bilinear', align_corners=False):
    return jt.nn.interpolate(
        input, size=size, scale_factor=scale_factor,
        mode=mode, align_corners=align_corners)
    
class DAFormerHead(nn.Module):
    def __init__(self,
                 in_channels, # 输入特征通道数列表: [64, 128, 320, 512]
                 in_index, # 输入特征索引列表: [0, 1, 2, 3]
                 channels=256, # 融合后特征通道数: 256
                 dropout_ratio=0.1, # Dropout 概率
                 num_classes=19, # 分割类别数
                 norm_cfg=dict(type='BN'), # 归一化配置
                 align_corners=False, # 双线性插值对齐方式
                 decoder_params=None): # 解码器参数
        super().__init__()
        self.in_channels = in_channels
        self.in_index = in_index
        self.channels = channels
        self.num_classes = num_classes
        self.align_corners = align_corners

        # 解析 decoder_params
        embed_dims = decoder_params['embed_dims'] # 嵌入维度
        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * len(self.in_index) # 扩展为列表
        embed_cfg = decoder_params['embed_cfg']
        embed_neck_cfg = decoder_params['embed_neck_cfg']
        fusion_cfg = decoder_params['fusion_cfg']

        # 为 ASPP 配置添加 align_corners
        fusion_cfg['align_corners'] = align_corners

        # 构建嵌入层：统一各尺度特征的通道数
        embed_layers = []
        for i, in_ch, embed_dim in zip(self.in_index, self.in_channels, embed_dims):
            if i == self.in_index[-1]:
                # 最高层特征使用 embed_neck_cfg
                embed_layers.append(build_layer(in_ch, embed_dim, **embed_neck_cfg))
            else:
                # 其他层使用 embed_cfg
                embed_layers.append(build_layer(in_ch, embed_dim, **embed_cfg))
        # Jittor 没有 nn.ModuleDict，所以用 ModuleList 并配合 in_index 来访问
        self.embed_layers = nn.ModuleList(embed_layers)

        # 融合层：使用 Sep-ASPP 融合多尺度特征
        self.fuse_layer = build_layer(
            sum(embed_dims), self.channels, **fusion_cfg)
        
        # Dropout
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        # 分类头：1×1 卷积
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
    
    def execute(self, inputs):
        x = inputs
        n = x[0].shape[0]

        # 获取 F1 的尺寸作为上采样目标
        # Jittor 不能用 size 而应该用 shape 来获取尺寸
        os_size = x[0].shape[2:] # (H/4, W/4)
        _c = {} # 临时存储嵌入后的特征

        # Step 1 & 2: 嵌入并上采样到最高分辨率
        for idx, i in enumerate(self.in_index):
            # 通过嵌入层统一通道数
            _c[i] = self.embed_layers[idx](x[i]) # (B, H*W, embed_dim)
            # 转换形状
            _c[i] = _c[i].permute(0, 2, 1)  # (B, C, H*W)
            _c[i] = _c[i].reshape(n, -1, x[i].shape[2], x[i].shape[3])  # (B, C, H, W)
            # 上采样到最高分辨率
            if _c[i].shape[2:] != os_size:
                _c[i] = resize(
                    _c[i], size=os_size,
                    mode='bilinear', align_corners=self.align_corners)
        
        # Step 3: 拼接所有尺度的特征
        x = jt.concat(list(_c.values()), dim=1)  # (B, sum(embed_dims), H/4, W/4)

        # Step 4: 使用 Sep-ASPP 融合
        x = self.fuse_layer(x)  # (B, channels, H/4, W/4)

        # Step 5: Dropout + 分类
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv_seg(x)  # (B, num_classes, H/4, W/4)

        return x
