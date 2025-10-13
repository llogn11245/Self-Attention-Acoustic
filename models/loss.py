import torch
import torch.nn as nn
import torch.nn.functional as F

class CELoss(nn.Module):
    def __init__(self, ignore_index=None, reduction='mean'):
        super(CELoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits (Tensor): Logits shape (N, C)/(N,T,C)
        """
        if logits.dim() == 3:
            logits = logits.transpose(1,2)
        
        # print(logits.shape, targets.shape)
        loss = F.cross_entropy(
            logits,
            targets,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )
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
        print("Using Cross Entropy Loss")
        return CELoss(
            ignore_index= config['blank'],
            reduction= config['reduction'],
            label_smoothing= config['label_smoothing']
        )
    elif config['type'] == 'kldiv_loss':
        print("Using KL Divergence Loss")
        return KLDivLoss(reduction= config['reduction'])