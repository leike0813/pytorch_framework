import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections.abc import Sequence
from typing import Optional


class OrderedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, class_values=[], weight=None, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.CELoss = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing, reduction='none')
        assert isinstance(num_classes, int) and num_classes > 2, 'num_classes must be an integer value greater than 2'
        self.num_classes = num_classes
        assert reduction in ('mean', 'sum', 'none'), f'Invalid reduction method {reduction}'
        self.reduction = reduction
        assert isinstance(class_values, Sequence), 'class_values must be a sequence'
        if len(class_values) == 0:
            self.register_buffer('class_values', torch.tensor([i for i in range(num_classes)], dtype=torch.int64))
            self._map_class_values = False
        else:
            assert len(class_values) == self.num_classes, 'length of class_values must be coincident with number of classes'
            _inc = [class_values[i] > class_values[i - 1] for i in range(1, self.num_classes)]
            _dec = [class_values[i] < class_values[i - 1] for i in range(1, self.num_classes)]
            assert np.all(_inc) or np.all(_dec), 'class_values must be strictly monotonically increasing or decreasing'
            self.register_buffer('class_values', torch.tensor(class_values, dtype=torch.int64).to('cuda' if torch.cuda.is_available() else 'cpu'))
            self._map_class_values = True
        self.class_values: Tensor
        self.value_diff = torch.abs(self.class_values[-1] - self.class_values[0]).item()

    def forward(self, input, target):
        ce_loss = self.CELoss(input, target)
        class_diff = torch.abs(
            torch.gather(input=self.class_values, dim=0, index=torch.argmax(input, dim=1))
            - torch.gather(input=self.class_values, dim=0, index=target)
        ) if self._map_class_values else torch.abs(
            torch.argmax(input, dim=1) - target
        )
        multiplier = 1. + class_diff / self.value_diff
        loss = ce_loss * multiplier
        if self.reduction == 'sum':
            loss = loss.sum()
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss


if __name__ == '__main__':
    oce = OrderedCrossEntropyLoss(4)
    input = torch.rand((8, 4))
    target = torch.tensor((2, 1, 0, 3, 3, 0, 0, 2))
    loss = oce(input, target)

    ii = 0
