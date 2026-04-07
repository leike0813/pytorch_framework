import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionLoss(nn.Module):
    def __init__(self, reduction='mean', normalize=True):
        """
        (归一化的)位置损失函数

        参数:
            reduction: 损失归约方式 ('mean', 'sum', 'none')
            normalize: 是否根据序列长度归一化损失
        """
        super().__init__()
        self.reduction = reduction
        self.normalize = normalize

    def forward(self, logits_or_prob, target_positions, seq_lengths=None):
        """
        计算归一化的位置损失

        参数:
            logits: 模型输出的logits张量 (batch_size, seq_len)
            target_positions: 真实变点位置 (batch_size,)
            seq_lengths: 序列实际长度 (batch_size,)

        返回:
            归一化的位置损失值
        """
        # 确保输入张量在相同设备上
        device = logits_or_prob.device
        target_positions = target_positions.to(device)
        if seq_lengths is not None:
            seq_lengths = seq_lengths.to(device)

        batch_size, max_len = logits_or_prob.shape

        # 1. 计算预测位置（期望值）
        # 创建位置索引 [0, 1, ..., max_len-1]
        positions = torch.arange(max_len, device=device, dtype=torch.float32)

        if logits_or_prob.min() < 0 or logits_or_prob.max() > 1: # 仍为logits，需要转换为概率分布
            logits_or_prob = F.softmax(logits_or_prob, dim=1)

        # 计算预测位置：Σ(position * probability)
        predicted_positions = torch.sum(positions.unsqueeze(0) * logits_or_prob, dim=1)

        # 2. 计算绝对距离损失
        # L1损失：|预测位置 - 真实位置|
        abs_diff = torch.abs(predicted_positions - target_positions.float())

        # 3. 根据序列有效长度归一化
        if self.normalize:
            # 转换为有效长度比例 (0-1范围)
            if seq_lengths is not None:
                normalized_diff = abs_diff / seq_lengths.float()
            else:
                normalized_diff = abs_diff / max_len
            loss = normalized_diff
        else:
            loss = abs_diff

        # 4. 处理归约方式
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

    def get_predicted_positions(self, logits_or_prob):
        """辅助函数：获取预测位置（不计算梯度）"""
        with torch.no_grad():
            _, max_len = logits_or_prob.shape
            positions = torch.arange(max_len, device=logits_or_prob.device, dtype=torch.float32)
            if logits_or_prob.min() < 0 or logits_or_prob.max() > 1:  # 仍为logits，需要转换为概率分布
                logits_or_prob = F.softmax(logits_or_prob, dim=1)
            return torch.sum(positions.unsqueeze(0) * logits_or_prob, dim=1)


if __name__ == '__main__':
    pred = torch.Tensor([[-0.34, -0.22, -0.02, 0.13, 0.63, 3.56, 1.52, 0.44],
                         [0.21, 0.44, 1.14, 3.92, 1.88, -0.32, -2.71, -4.41]])
    target = torch.LongTensor([2, 5])
    loss_fn = PositionLoss()
    loss = loss_fn(pred, target)

    loss2 = loss_fn(pred, target, seq_lengths=torch.LongTensor([7, 6]))

    ii = 0