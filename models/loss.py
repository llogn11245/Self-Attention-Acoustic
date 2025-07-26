import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=4, reduction="mean"):
        """
        Cross Entropy Loss cho bài toán sequence labeling
        
        Args:
            ignore_index (int): Chỉ số của các phần tử cần bỏ qua (thường dùng cho padding)
            reduction (str): Phương thức giảm kích thước ('mean', 'sum', 'none')
        """
        super(CrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets, input_lengths=None, target_lengths=None):
        # Chuyển đổi kích thước logits để phù hợp với CrossEntropyLoss
        # [B, T, C] -> [B, C, T] (theo yêu cầu của nn.CrossEntropyLoss)
        logits = logits.transpose(1, 2)  # [batch_size, vocab_size, sequence_length]
        
        # Tính loss
        loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index,
            reduction=self.reduction
        )
        loss = loss_fn(logits, targets)
        
        return loss