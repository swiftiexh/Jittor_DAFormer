# Backbone: MiT-B5

from functools import partial
from jittor import nn
import jittor as jt
from jittor import init
import math


# 重叠补丁合并层
class OverlapPatchEmbed(nn.Module):
    def __init__(self,
                 patch_size,
                 stride,
                 in_chans,
                 embed_dim):
        super().__init__()
        # 使用卷积实现重叠的 Patch Embedding
        # 实现了下采样：Stage1 下采样 4 倍，Stage2/3/4 下采样 2 倍
        # 512 - 128 - 64 - 32 - 16
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size // 2, patch_size // 2) 
        )
        # 归一化层
        self.norm = nn.LayerNorm(embed_dim)

    def execute(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        # Jittor 自动处理内存布局，不需要像 PyTorch 那样调用 contiguous()，直接 reshape 和 transpose 就行
        x = x.flatten(2).transpose(1, 2) # (B, C, H, W) -> (B, H*W, C)
        x = self.norm(x) # 归一化：(B, H*W, C)
        return x, H, W


# 多头注意力层
class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5 # 缩放因子

        # nn.Linear 对输入的要求是「最后一维等于 in_features」。它接受任意形状，只在最后一维上做线性映射
        # nn.Conv2d(in_channels, out_channels, ...)求输入是 4D 张量，形状为 (N, C_in, H, W)，其中 C_in 必须等于 in_channels。
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        # 序列缩减策略
        # sr_ratio（spatial reduction ratio）在注意力模块中表示对 k / v 序列进行空间缩减的比例
        # 用于减少序列长度 N，从而降低注意力计算量与内存消耗
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
    
    def execute(self, x, H, W):
        B, N, C = x.shape
        # q: (B, num_heads, N, C//num_heads)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # 序列缩减策略
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W) # (B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)  # (B, N', C)
            x_ = self.norm(x_) # (B, N', C)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # (2, B, num_heads, N', C//num_heads)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # (2, B, num_heads, N, C//num_heads)
        k, v = kv[0], kv[1] # (B, num_heads, N(or N') , C//num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, num_heads, N(or N'), N(or N'))
        attn = jt.nn.softmax(attn, dim=-1) # 注意力权重
        attn = self.attn_drop(attn) # Dropout

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # (B, N, C)
        x = self.proj(x) # 线性映射
        x = self.proj_drop(x) # Dropout

        return x

# 深度卷积
class DWConv(nn.Module):    
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        # 对每个输入通道单独做一个卷积
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def execute(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W) # (B, C, H, W)
        x = self.dwconv(x) # 深度卷积
        x = x.flatten(2).transpose(1, 2) # (B, H*W, C)
        return x

class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # 两层全连接 + 激活 + Dropout
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features) # 深度可分离卷积
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def execute(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self,
                 dim, # 阶段维度
                 num_heads, # 注意力头数
                 mlp_ratio=4., # MLP 扩展比例
                 qkv_bias=False, # 使用偏置
                 qk_scale=None, # qk 缩放因子
                 drop=0., # Dropout 概率
                 attn_drop=0., # 注意力 Dropout 概率
                 drop_path=0., # Stochastic Depth 概率
                 act_layer=nn.GELU, # 激活函数
                 norm_layer=nn.LayerNorm, # 归一化层
                 sr_ratio=1): # 空间缩放比例
        super().__init__()
        # 多头注意力层
        self.norm1 = norm_layer(dim) # 归一化层 
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio)
        # 随机深度的残差连接
        # 参考自 nk_seg_main
        self.drop_path = nn.DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # MLP 层
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)
    
    def execute(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class MixVisionTransformer(nn.Module):
    def __init__(self,
                 in_chans=3, # 输入通道数
                 embed_dims=[64, 128, 320,512], # 每个阶段的嵌入维度
                 num_heads=[1, 2, 5, 8], # 每个阶段的注意力头数
                 mlp_ratios=[4, 4, 4, 4], # MLP 扩展比例
                 qkv_bias=False, # 使用偏置
                 qk_scale=None, # qk 缩放因子
                 drop_rate=0., # Dropout 概率
                 attn_drop_rate=0.0, # 注意力 Dropout 概率
                 drop_path_rate=0.1, # Stochastic Depth 概率
                 norm_layer=nn.LayerNorm, # 归一化层
                 depths=[3, 6, 40, 3], # 每个阶段的 Transformer 层数
                 sr_ratios=[8, 4, 2, 1], # 每个阶段的空间缩放比例
                 pretrained=None, # 预训练权重路径
                 freeze_patch_embed=False): # 是否冻结 Patch 嵌入层
        super().__init__()
        self.pretrained = pretrained
        self.depths = depths
        # Patch Embedding 层：输出的尺寸 (B, H*W, C)
        self.patch_embed1 = OverlapPatchEmbed(
            patch_size=7,
            stride=4,
            in_chans=in_chans,
            embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(
            patch_size=3,
            stride=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3])
        if freeze_patch_embed:
            self.freeze_patch_emb()

        # Stochastic Depth 比例
        dpr = [x.item() for x in jt.linspace(0, drop_path_rate, sum(depths))]

        # 构建每个阶段的 Transformer 块
        cur = 0
        self.block1 = nn.ModuleList([
            Block(
                dim=embed_dims[0], # 阶段维度
                num_heads=num_heads[0], # 注意力头数
                mlp_ratio=mlp_ratios[0], # MLP 扩展比例
                qkv_bias=qkv_bias, # 使用偏置
                qk_scale=qk_scale, # qk 缩放因子
                drop=drop_rate, # Dropout 概率
                attn_drop=attn_drop_rate, # 注意力 Dropout 概率
                drop_path=dpr[cur + i], # Stochastic Depth 概率
                norm_layer=norm_layer, # 归一化层
                sr_ratio=sr_ratios[0] # 空间缩放比例
            ) for i in range(depths[0])
        ])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([
            Block(
                dim=embed_dims[1],
                num_heads=num_heads[1],
                mlp_ratio=mlp_ratios[1],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[1]) for i in range(depths[1])
        ])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([
            Block(
                dim=embed_dims[2],
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[2]) for i in range(depths[2])
        ])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([
            Block(
                dim=embed_dims[3],
                num_heads=num_heads[3],
                mlp_ratio=mlp_ratios[3],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[3]) for i in range(depths[3])
        ])
        self.norm4 = norm_layer(embed_dims[3])

        # Jittor 没有 apply 函数，我们在外部调用 init_weights 方法来初始化权重
    
    # 冻结 Patch Embedding 层的参数
    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    # 权重初始化
    def init_weights(self):
        if self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.trunc_normal_(m.weight, mean=0., std=.02, a=-0.04, b=0.04)
                    if m.bias is not None:
                        init.constant_(m.bias, 0.)
                elif isinstance(m, nn.LayerNorm):
                    init.constant_(m.weight, 1.0)
                    if m.bias is not None:
                        init.constant_(m.bias, 0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    init.gauss_(m.weight, mean=0., std=math.sqrt(2.0 / fan_out))
                    if m.bias is not None:
                        init.constant_(m.bias, 0.)
        elif isinstance(self.pretrained, str):
            self.load_parameters(jt.load(self.pretrained))

    # 前向传播
    def execute(self, x):
        B = x.shape[0]
        outs = []

        # Stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x)

        # Stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x)

        # Stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x)

        # Stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x)

        return outs

    

def mit_b5(pretrained=None, **kwargs):
    model = MixVisionTransformer(
        embed_dims=[64, 128, 320, 512], # 每个阶段的嵌入维度 
        num_heads=[1, 2, 5, 8], # 每个阶段的注意力头数 
        mlp_ratios=[4, 4, 4, 4], # MLP 扩展比例 
        qkv_bias=True, # 使用偏置 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), # 归一化层 
        depths=[3, 6, 40, 3], # 每个阶段的 Transformer 层数 
        sr_ratios=[8, 4, 2, 1], # 每个阶段的空间缩放比例 
        drop_rate=0.0, # Dropout 概率  
        drop_path_rate=0.1, # Stochastic Depth 概率 
        pretrained=pretrained, # 预训练权重路径 
        **kwargs)
    model.init_weights()
    return model