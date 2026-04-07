import torch


__all__ = [
    'gaussian_one_hot'
]


def gaussian_one_hot(targets, num_classes=-1, sigma=0.5):
    batch_size = targets.size(0)
    if num_classes == 1:
        num_classes = targets.max() + 1

    # 创建位置索引矩阵 [0, 1, ..., num_classes - 1]
    positions = torch.arange(num_classes).expand(batch_size, -1).float().to(targets.device)

    # 将目标位置扩展为矩阵
    target_pos = targets.unsqueeze(1).expand(-1, num_classes).float()

    # 计算距离矩阵 (绝对距离)
    dist_matrix = torch.abs(positions - target_pos)

    # ===== 1. 创建概率扩散的目标分布 =====
    # 高斯扩散核 (避免数值下溢使用对数空间计算)
    log_gaussian = -0.5 * (dist_matrix / sigma) ** 2
    targets = torch.exp(log_gaussian)

    return targets


if __name__ == '__main__':
    targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(gaussian_one_hot(targets, num_classes=10, sigma=0.5))