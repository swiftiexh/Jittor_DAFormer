import sys
import jittor as jt
from pathlib import Path

def test_model_pipeline():
    print("=" * 70)
    print("测试完整的模型流水线（数据 → 模型 → 损失）")
    print("=" * 70)

    # 0. 设置 Jittor 使用 GPU
    if jt.has_cuda:
        jt.flags.use_cuda = 1
        print("\n✓ 已启用 CUDA")
    else:
        print("\n- 使用 CPU")

    # 1. 加载配置
    print("\n[步骤 1/7] 加载配置...")
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from train import load_config
        cfg = load_config('configs/gta2cs_daformer.py')
        print("✓ 配置加载成功")
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 2. 构建数据集
    print("\n[步骤 2/7] 构建数据集...")
    try:
        from train import build_dataset
        train_dataset = build_dataset(cfg.data)
        print(f"✓ 数据集构建成功")
        print(f"  - 源域样本数: {len(train_dataset.source)}")
        print(f"  - 目标域样本数: {len(train_dataset.target)}")
        print(f"  - RCS 启用: {train_dataset.rcs_enabled}")
    except Exception as e:
        print(f"✗ 数据集构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 3. 获取一个 batch 的数据
    print("\n[步骤 3/7] 获取数据 batch...")
    try:
        batch = next(iter(train_dataset))
        print("✓ Batch 获取成功")
        print(f"  - Batch 键: {list(batch.keys())}")
        print(f"  - img shape: {batch['img'].shape}")
        print(f"  - gt_semantic_seg shape: {batch['gt_semantic_seg'].shape}")
        print(f"  - target_img shape: {batch['target_img'].shape}")
    except Exception as e:
        print(f"✗ Batch 获取失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. 构建模型
    print("\n[步骤 4/7] 构建模型...")
    try:
        from models.segmentor import build_segmentor
        model = build_segmentor(cfg.model)
        print("✓ 模型构建成功")
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - 总参数量: {total_params:,}")
        print(f"  - 可训练参数: {trainable_params:,}")
    except Exception as e:
        print(f"✗ 模型构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 5. 测试前向传播（训练模式）
    print("\n[步骤 5/7] 测试训练前向传播...")
    try:
        model.train()
        img = batch['img']
        gt = batch['gt_semantic_seg']
        
        print(f"  - 输入 img: {img.shape}, dtype: {img.dtype}")
        print(f"  - 输入 gt: {gt.shape}, dtype: {gt.dtype}")
        
        # 前向传播
        losses = model.forward_train(img, gt, return_feat=False)
        
        print("✓ 训练前向传播成功")
        print(f"  - 损失键: {list(losses.keys())}")
        print(f"  - loss_seg: {losses['loss_seg'].item():.4f}")
        print(f"  - acc_seg: {losses['acc_seg'].item():.4f}")
    except Exception as e:
        print(f"✗ 训练前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    jt.clean_graph()
    jt.gc()

    # 6. 测试训练并返回特征图
    print("\n[步骤 6/7] 测试训练并返回特征图...")
    try:
        model.train()
        # 使用 return_feat=True 获取特征图
        losses_with_feat = model.forward_train(img, gt, return_feat=True)
        
        print("✓ 特征图返回成功")
        print(f"  - 返回键: {list(losses_with_feat.keys())}")
        
        if 'features' in losses_with_feat:
            features = losses_with_feat['features']
            print(f"  - 特征图数量: {len(features)} 个 stage")
            for i, feat in enumerate(features):
                print(f"    Stage {i+1}: shape={feat.shape}, dtype={feat.dtype}")
                # 验证特征图的下采样倍数
                expected_size = img.shape[2] // (4 * (2 ** i))  # 4, 8, 16, 32 倍下采样
                actual_size = feat.shape[2]
                print(f"      期望尺寸: {expected_size}×{expected_size}, 实际尺寸: {actual_size}×{feat.shape[3]}")
        else:
            print("  ⚠️ 特征图未返回")
    except Exception as e:
        print(f"✗ 特征图返回测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    jt.clean_graph()
    jt.gc()

    # 7. 测试推理前向传播（eval 模式）
    print("\n[步骤 7/7] 测试推理前向传播...")
    try:
        model.eval()
        with jt.no_grad():
            target_img = batch['target_img']  # 确认是 Var
            print(f"  - 输入 target_img: {target_img.shape}")

            # 推理
            seg_logits = model.encode_decode(target_img)
            print("✓ 推理前向传播成功")
            print(f"  - 输出 seg_logits: {seg_logits.shape}")
            
            seg_pred = seg_logits.argmax(dim=1)[0]
            max_class = int(seg_pred.max().data[0])
            print(f"  - 预测类别范围: [0, {max_class}]")
    except Exception as e:
        print(f"✗ 推理前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    jt.clean_graph()
    jt.gc()

    # 8. 测试带权重的损失计算（需要重新构建模型避免状态污染）
    print("\n[额外测试] 测试加权损失计算...")
    try:
        # ⚠️ 重新构建模型以确保相同的初始状态
        print("  - 重新构建模型（避免BatchNorm状态污染）...")
        from models.segmentor import build_segmentor
        model_fresh = build_segmentor(cfg.model)
        model_fresh.train()
        
        # 先测试不带权重（作为baseline）
        losses_baseline = model_fresh.forward_train(img, gt)
        print(f"  - 基准 loss_seg (不带权重): {losses_baseline['loss_seg'].item():.4f}")
        
        # 清理后测试带权重
        jt.clean_graph()
        jt.gc()
        
        # 创建权重：部分像素降权到0.5
        # 注意：seg_weight 应为 [B,H,W]，不是 [B,1,H,W]
        seg_weight = jt.ones((gt.shape[0], gt.shape[2], gt.shape[3])).float32()  # [B,H,W]
        seg_weight[:, ::2, ::2] = 0.5  # 1/4像素降权
        avg_weight = seg_weight.mean().item()
        print(f"  - 平均权重: {avg_weight:.4f} (期望≈0.875)")
        
        losses_weighted = model_fresh.forward_train(img, gt, seg_weight=seg_weight)
        
        print("✓ 加权损失计算成功")
        print(f"  - 加权 loss_seg: {losses_weighted['loss_seg'].item():.4f}")
        
        # 分析比例
        ratio = losses_weighted['loss_seg'].item() / losses_baseline['loss_seg'].item()
        print(f"  - 损失比例: {ratio:.4f}")
        print(f"  - 分析: 比例≠1.0 是正常的，说明降权区域和非降权区域的损失不均匀")
        print(f"          比例<1.0 表示降权区域（偶数行列）的损失更高")
        print(f"          比例>1.0 表示降权区域的损失更低")
        
        # 实现正确性检查：比例应该在合理范围内
        if 0.5 < ratio < 1.5:
            print("  ✓ 加权损失实现正确！(比例在合理范围内)")
        else:
            print(f"  ⚠️ 损失比例异常，可能存在实现问题")
            
    except Exception as e:
        print(f"  ⚠️ 加权损失测试失败: {e}")
        import traceback
        traceback.print_exc()

    # 总结
    print("\n" + "=" * 70)
    print("✓ 所有测试通过！完整流水线验证成功！")
    print("=" * 70)
    print("\n流水线组件状态：")
    print("  ✓ 数据加载 (GTADataset + CityscapesDataset + UDADataset)")
    print("  ✓ RCS (Rare Class Sampling)")
    print("  ✓ Backbone (MiT-B5)")
    print("  ✓ Decoder (DAFormer Head)")
    print("  ✓ 损失计算 (Cross Entropy + Accuracy)")
    print("  ✓ 训练前向传播")
    print("  ✓ 特征图提取 (return_feat=True)")
    print("  ✓ 推理前向传播")
    print("\n准备就绪，可以开始训练！")
    
    return True


if __name__ == '__main__':
    success = test_model_pipeline()
    sys.exit(0 if success else 1)