import torch
import torch.nn as nn
import torch.nn.functional as F

class CELoss(nn.Module):
    def __init__(self, ignore_index=4, reduction="mean", label_smoothing=0.1):
        """
        Cross Entropy Loss cho bài toán sequence labeling
        
        Args:
            ignore_index (int): Chỉ số của các phần tử cần bỏ qua (thường dùng cho padding)
            reduction (str): Phương thức giảm kích thước ('mean', 'sum', 'none')
        """
        super(CELoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets, input_lengths=None, target_lengths=None):
        # Chuyển đổi kích thước logits để phù hợp với CrossEntropyLoss
        # [B, T, C] -> [B, C, T] (theo yêu cầu của nn.CrossEntropyLoss)
        logits = logits.transpose(1, 2)  # [batch_size, vocab_size, sequence_length]
        
        # Tính loss
        loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing
        )
        loss = loss_fn(logits, targets)
        
        return loss

class KLDivLoss(nn.Module):
    def __init__(self, reduction="batchmean"):
        """
        KL Divergence Loss cho bài toán sequence labeling
        
        Args:
            reduction (str): 'mean', 'sum', 'batchmean', 'none' 
        """
        super(KLDivLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, targets, input_lengths=None, target_lengths=None):
        logits = F.log_softmax(logits, dim= 2)
        target_probs = torch.zeros_like(logits).scatter_(2, targets.unsqueeze(2), 1.0)

        # Tính KL Divergence
        loss_kldiv = nn.KLDivLoss(reduction=self.reduction)

        loss = loss_kldiv(logits, target_probs)
        
        return loss
    
class CTCLoss(nn.Module):
    def __init__(self, blank=0, reduction="mean"):
        super(CTCLoss, self).__init__()
        self.blank = blank
        self.reduction = reduction
        self.ctc_loss_fn = nn.CTCLoss(
            blank=self.blank,
            reduction=self.reduction,
        )

    def forward(self, logits, targets, input_lengths, target_lengths):
        log_probs = F.log_softmax(logits, dim=-1)

        # [B, T, C] -> [T, B, C] 
        log_probs = log_probs.transpose(0, 1)  # [sequence_length, batch_size, vocab_size]
        
        loss = self.ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
        
        return loss
    
def build_loss(config):
    if config['type'] == 'ce_loss':
        return CELoss(
            ignore_index= config['blank'],
            reduction= config['reduction'],
            label_smoothing= config['label_smoothing']
        )
    elif config['type'] == 'kldiv_loss':
        return KLDivLoss(reduction= config['reduction'])