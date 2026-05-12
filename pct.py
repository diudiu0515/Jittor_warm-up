#!/usr/bin/env python
"""
PCT (Point Cloud Transformer) for ModelNet40 Classification - 示例代码

本代码提供了基于 Jittor 框架的 PCT 模型，用于 ModelNet40 三维形状分类任务。
选手需要完成标注为 TODO 的部分，训练模型并生成测试集预测结果 result.json。

Usage:
    python pct.py

依赖安装:
    pip install jittor
"""

import os
import json
import math
import time
import argparse
import numpy as np
import jittor as jt
from jittor import nn
from jittor.dataset import Dataset


# ============================================================
# 数据集
# ============================================================

class ModelNet40Dataset(Dataset):
    """ModelNet40 点云数据集。

    加载预处理好的 npy 文件。
    - 训练集: data/train_points.npy + data/train_labels.npy
    - 测试集: data/test_points.npy (无标签)
    """

    def __init__(self, data_dir='./data', split='train', n_points=1024,
                 augment=False, batch_size=32, shuffle=False, num_workers=4):
        super().__init__(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.n_points = n_points
        self.augment = augment
        self.split = split

        pts_path = os.path.join(data_dir, f'{split}_points.npy')
        assert os.path.exists(pts_path), f"{pts_path} not found."

        self.point_clouds = np.load(pts_path)  # (N, 2048, 3)
        self.n_cached = self.point_clouds.shape[1]

        if split == 'train':
            lbl_path = os.path.join(data_dir, f'{split}_labels.npy')
            assert os.path.exists(lbl_path), f"{lbl_path} not found."
            self.labels = np.load(lbl_path)  # (N,)
        else:
            self.labels = None  # 测试集不提供标签

        self.total_len = len(self.point_clouds)
        self.set_attrs(total_len=self.total_len)

    def __getitem__(self, idx):
        pts = self.point_clouds[idx]
        replace = self.n_cached < self.n_points
        choice = np.random.choice(self.n_cached, self.n_points, replace=replace)
        points = pts[choice].copy()

        if self.augment:
            # TODO: 实现数据增强策略
            # 提示：可以考虑随机旋转、随机缩放、随机抖动等
            # 示例：随机绕 Y 轴旋转
            theta = np.random.uniform(0, 2 * np.pi)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            R = np.array([[cos_t, 0, sin_t],
                          [0, 1, 0],
                          [-sin_t, 0, cos_t]], dtype=np.float32)
            points = points @ R.T

            scale = np.random.uniform(0.8, 1.25)
            points *= scale

            shift = np.random.uniform(-0.1, 0.1, (1, 3))
            points += shift

            noise = np.random.normal(0, 0.01, points.shape)
            noise = np.clip(noise, -0.05, 0.05)
            points += noise

        if self.labels is not None:
            return points.astype(np.float32), self.labels[idx]
        else:
            return points.astype(np.float32), idx  # 测试集返回索引


# ============================================================
# PCT 模型
# ============================================================

class SA_Layer(nn.Module):
    """Self-Attention layer for PCT."""

    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def execute(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)  # (B, N, C//4)
        x_k = self.k_conv(x)                     # (B, C//4, N)
        x_v = self.v_conv(x)                     # (B, C, N)
        energy = jt.nn.bmm(x_q, x_k)            # (B, N, N)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = jt.nn.bmm(x_v, attention)          # (B, C, N)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class PCT(nn.Module):
    """Point Cloud Transformer for classification.

    Input:  (B, 3, N) point cloud
    Output: (B, num_classes) logits
    """

    def __init__(self, num_classes=40):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 128, 1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        self.sa1 = SA_Layer(128)
        self.sa2 = SA_Layer(128)
        self.sa3 = SA_Layer(128)
        self.sa4 = SA_Layer(128)

        self.conv_fuse = nn.Sequential(
            nn.Conv1d(512, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(scale=0.2))

        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)

    def execute(self, x):
        B, _, N = x.shape
        x = nn.relu(self.bn1(self.conv1(x)))
        x = nn.relu(self.bn2(self.conv2(x)))

        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)

        x = jt.concat([x1, x2, x3, x4], dim=1)  # (B, 512, N)
        x = self.conv_fuse(x)                      # (B, 1024, N)
        x = jt.max(x, dim=2)                       # (B, 1024)

        x = nn.relu(self.bn_fc1(self.fc1(x)))
        x = self.dp1(x)
        x = nn.relu(self.bn_fc2(self.fc2(x)))
        x = self.dp2(x)
        x = self.fc3(x)
        return x


# ============================================================
# 学习率调度器
# ============================================================

class CosineAnnealingLR:
    def __init__(self, optimizer, T_max, eta_min=1e-5):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        lr = self.eta_min + (self.base_lr - self.eta_min) * \
             (1 + math.cos(math.pi * self.current_epoch / self.T_max)) / 2
        self.optimizer.lr = lr
        return lr


# ============================================================
# 训练与推理
# ============================================================

def train_one_epoch(model, train_loader, optimizer, epoch, log_interval=20):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    t0 = time.time()

    for batch_idx, (points, labels) in enumerate(train_loader):
        points = jt.array(points).permute(0, 2, 1)  # (B, 3, N)
        labels = jt.array(labels).reshape(-1)

        logits = model(points)
        loss = nn.cross_entropy_loss(logits, labels)

        optimizer.step(loss)

        preds = logits.argmax(dim=1)[0]
        total_correct += (preds == labels).sum().item()
        total_count += labels.shape[0]
        total_loss += loss.item() * labels.shape[0]

        if (batch_idx + 1) % log_interval == 0:
            print(f"  Epoch [{epoch}] Batch [{batch_idx+1}] "
                  f"Loss: {total_loss/total_count:.4f}  "
                  f"Acc: {total_correct/total_count*100:.2f}%  "
                  f"Time: {time.time()-t0:.1f}s")

    return total_loss / total_count, total_correct / total_count * 100


def predict(model, test_loader):
    """对测试集进行预测，返回 {样本编号: 预测类别} 字典。"""
    model.eval()
    results = {}

    with jt.no_grad():
        for points, indices in test_loader:
            points = jt.array(points).permute(0, 2, 1)
            indices = jt.array(indices).reshape(-1)

            logits = model(points)
            preds = logits.argmax(dim=1)[0]

            for i in range(preds.shape[0]):
                sample_id = int(indices[i].item())
                results[str(sample_id)] = int(preds[i].item())

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--n_points', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    jt.flags.use_cuda = 1

    print("=" * 60)
    print("ModelNet40 Classification - PCT Baseline")
    print(f"Points: {args.n_points}  Batch: {args.batch_size}  "
          f"Epochs: {args.epochs}  LR: {args.lr}")
    print("=" * 60)

    # --------------------------------------------------
    # 加载数据
    # --------------------------------------------------
    train_loader = ModelNet40Dataset(
        data_dir=args.data_dir, split='train', n_points=args.n_points,
        augment=True, batch_size=args.batch_size, shuffle=True)
    test_loader = ModelNet40Dataset(
        data_dir=args.data_dir, split='test', n_points=args.n_points,
        augment=False, batch_size=args.batch_size, shuffle=False)

    print(f"Train: {train_loader.total_len} samples")
    print(f"Test:  {test_loader.total_len} samples")

    # --------------------------------------------------
    # 构建模型
    # --------------------------------------------------
    model = PCT(num_classes=40)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    # --------------------------------------------------
    # TODO: 设置优化器和学习率调度器
    # 提示：可以尝试 SGD / Adam，配合 cosine annealing 等调度策略
    # --------------------------------------------------
    # optimizer = nn.SGD(model.parameters(), lr=args.lr,
    #                    momentum=0.9, weight_decay=1e-4)
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    optimizer = nn.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    # --------------------------------------------------
    # 训练循环
    # --------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, epoch)
        lr = scheduler.step()
        print(f"Epoch [{epoch}/{args.epochs}]  "
              f"Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
              f"LR: {optimizer.lr:.6f}  Time: {time.time()-t0:.1f}s")

    # --------------------------------------------------
    # 保存模型
    # --------------------------------------------------
    model.save('pct_model.pkl')
    print("Model saved to pct_model.pkl")

    # --------------------------------------------------
    # 对测试集进行预测并保存 result.json
    # --------------------------------------------------
    print("Generating predictions on test set...")
    results = predict(model, test_loader)
    with open('result.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} predictions to result.json")
    print("Done!")


if __name__ == '__main__':
    main()
