import sys
import jittor as jt
import numpy as np
from pathlib import Path
from collections import Counter


def test_data_pipeline():
    """测试数据加载、pipeline、RCS 和 batch 构造（Jittor 版）"""
    print("=" * 60)
    print("开始验证数据部分...")
    print("=" * 60)

    # 0. 设置 Jittor 使用 GPU（如果可用）
    if jt.has_cuda:
        jt.flags.use_cuda = 1
        print("✓ 已启用 CUDA，Jittor 将自动在 GPU 上执行")
    else:
        print("- CUDA 不可用，使用 CPU")

    # 1. 加载配置
    print("\n[1/6] 加载配置...")
    sys.path.insert(0, str(Path(__file__).parent))
    from train import load_config, build_dataset

    try:
        cfg = load_config('configs/gta2cs_daformer.py')
        print("✓ 配置加载成功")
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        return False

    # 2. 构建数据集
    print("\n[2/6] 构建 UDA 数据集...")
    try:
        train_dataset = build_dataset(cfg.data)
        print(f"✓ 数据集构建成功")
        print(f"  - 源域样本数: {len(train_dataset.source)}")
        print(f"  - 目标域样本数: {len(train_dataset.target)}")
        print(f"  - RCS 启用: {train_dataset.rcs_enabled}")
        if train_dataset.rcs_enabled:
            print(f"  - RCS 类别数: {len(train_dataset.rcs_classes)}")
            print(f"  - RCS 类别: {train_dataset.rcs_classes[:5]}... (前5个)")
            print(f"  - 概率和: {train_dataset.rcs_classprob.sum():.4f}")
    except Exception as e:
        print(f"✗ 数据集构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 3. 测试单样本获取
    print("\n[3/6] 测试单样本获取...")
    try:
        sample = train_dataset[0]
        print(f"✓ 单样本获取成功")
        print(f"  - 样本键: {list(sample.keys())}")

        # 检查 img
        if 'img' in sample:
            img = sample['img']
            print(f"  - img 类型: {type(img)}")
            print(f"  - img shape: {img.shape}")
            print(f"  - img dtype: {img.dtype}")
            if isinstance(img, jt.Var):
                print(f"  - img 是 Jittor Var ✓")

        # 检查 gt_semantic_seg
        if 'gt_semantic_seg' in sample:
            gt = sample['gt_semantic_seg']
            print(f"  - gt_semantic_seg 类型: {type(gt)}")
            print(f"  - gt_semantic_seg shape: {gt.shape}")
            print(f"  - gt_semantic_seg dtype: {gt.dtype}")
            if isinstance(gt, jt.Var):
                print(f"  - gt_semantic_seg 是 Jittor Var ✓")
                gt_np = gt.numpy()
                unique_labels = np.unique(gt_np)
                print(f"  - gt 唯一值: {unique_labels.tolist()[:10]}... (前10个)")
                print(f"  - gt 最小值: {gt_np.min()}, 最大值: {gt_np.max()}")

        # 检查 target_img
        if 'target_img' in sample:
            tgt = sample['target_img']
            print(f"  - target_img 类型: {type(tgt)}")
            print(f"  - target_img shape: {tgt.shape}")
            print(f"  - target_img dtype: {tgt.dtype}")
            if isinstance(tgt, jt.Var):
                print(f"  - target_img 是 Jittor Var ✓")

    except Exception as e:
        print(f"✗ 单样本获取失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. 验证 RCS 采样效果
    print("\n[4/6] 验证 RCS 采样...")
    if train_dataset.rcs_enabled:
        try:
            print("  - RCS 类别概率分布（前5个稀有类）:")
            for i in range(min(5, len(train_dataset.rcs_classes))):
                cls = train_dataset.rcs_classes[i]
                prob = train_dataset.rcs_classprob[i]
                class_name = train_dataset.CLASSES[cls] if cls < len(train_dataset.CLASSES) else f"类{cls}"
                print(f"    类别 {cls} ({class_name}): {prob:.6f}")
            
            print("\n 统计被选为采样目标的稀有类（100次）:")
            target_classes = []
            for i in range(100):
                # 模拟 RCS 内部选择目标类别的过程
                c = np.random.choice(train_dataset.rcs_classes, p=train_dataset.rcs_classprob)
                target_classes.append(c)
            
            class_counter = Counter(target_classes)
            print("    前5个最常被选为目标的类别:")
            for cls, count in class_counter.most_common(5):
                class_name = train_dataset.CLASSES[cls] if cls < len(train_dataset.CLASSES) else f"类{cls}"
                print(f"      类别 {cls} ({class_name}): {count} 次 ({count}%)")
            
            # 单个样本示例
            s = train_dataset.get_rare_class_sample()
            print("\n  ✓ RCS 单样本示例，键:", list(s.keys()))
            if 'gt_semantic_seg' in s:
                gt = s['gt_semantic_seg']
                if isinstance(gt, jt.Var):
                    gt_np = gt.numpy()
                else:
                    gt_np = gt
                unique = np.unique(gt_np)
                print(f"    样本包含 {len(unique)} 个类别: {unique.tolist()}")
        except Exception as e:
            print(f"  ✗ RCS 验证失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  - RCS 未启用，跳过")

    # 5. 构建/使用 DataLoader（Jittor：dataset 已包含 set_attrs，可直接迭代）
    print("\n[5/6] 构建/获取 DataLoader...")
    try:
        train_loader = train_dataset 
        print(f"✓ DataLoader 准备就绪（使用 dataset 自身作为迭代器）")
        print(f"  - batch_size: {cfg.data['samples_per_gpu']}")
        print(f"  - num_workers: {cfg.data['workers_per_gpu']}")
    except Exception as e:
        print(f"✗ DataLoader 构建/准备失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 6. 测试 DataLoader 迭代和 batch 内容
    print("\n[6/6] 测试 DataLoader 和 Batch...")
    try:
        batch = next(iter(train_loader))
        print("✓ Batch 获取成功")
        print(f"  - Batch 键: {list(batch.keys())}")

        # 检查 source 数据
        print("\n[Source 数据]")
        if 'img' in batch:
            img = batch['img']
            print(f"  - img 类型: {type(img)}")
            print(f"  - img shape: {img.shape}")
            print(f"  - img dtype: {img.dtype}")
            if isinstance(img, jt.Var):
                print(f"  - img 是 Jittor Var (batch) ✓")
                img_np = img.numpy()
                print(f"  - img range: [{float(img_np.min()):.3f}, {float(img_np.max()):.3f}]")

        if 'gt_semantic_seg' in batch:
            gt = batch['gt_semantic_seg']
            print(f"  - gt_semantic_seg 类型: {type(gt)}")
            print(f"  - gt_semantic_seg shape: {gt.shape}")
            print(f"  - gt_semantic_seg dtype: {gt.dtype}")
            if isinstance(gt, jt.Var):
                print(f"  - gt_semantic_seg 是 Jittor Var (batch) ✓")
                gt_np = gt.numpy()
                unique_labels = np.unique(gt_np)
                print(f"  - 唯一标签: {unique_labels.tolist()}")

        # 检查 target 数据
        print("\n[Target 数据]")
        if 'target_img' in batch:
            tgt = batch['target_img']
            print(f"  - target_img 类型: {type(tgt)}")
            print(f"  - target_img shape: {tgt.shape}")
            print(f"  - target_img dtype: {tgt.dtype}")
            if isinstance(tgt, jt.Var):
                print(f"  - target_img 是 Jittor Var (batch) ✓")
                tgt_np = tgt.numpy()
                print(f"  - target_img range: [{float(tgt_np.min()):.3f}, {float(tgt_np.max()):.3f}]")

        # Jittor 自动管理 GPU，无需手动传输
        print("\n[GPU 执行验证]")
        if jt.flags.use_cuda == 1:
            print("  ✓ Jittor 已设置使用 CUDA，所有操作自动在 GPU 上执行")
            print("  - 无需手动 .cuda() 传输数据")
            print("  - 无需手动 .to(device) 传输模型")
        else:
            print("  - 当前使用 CPU 执行")

        print("\n" + "=" * 60)
        print("✓ 所有测试通过！数据部分验证完成（Jittor）")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n✗ Batch 测试失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_data_pipeline()
    sys.exit(0 if success else 1)