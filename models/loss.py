import torch
import torch.nn as nn
import torch.nn.functional as F

class CTCLoss(nn.Module):
    def __init__(self, blank=4, reduction="mean"):
        super(CTCLoss, self).__init__()
        self.blank = blank
        self.reduction = reduction
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=True)

    def forward(self, logits, targets, input_lengths, target_lengths):
        log_probs = F.log_softmax(logits, dim=-1)  # [B, T, Vocab]
        
        log_probs = log_probs.transpose(0, 1)  # Now [T, B, Vocab]
        
        input_lengths = input_lengths
        target_lengths = target_lengths
        
        # Tính toán CTC loss
        loss = self.ctc_loss(
            log_probs,          # [T, B, Vocab]
            targets,            # [B, U]
            input_lengths,      # [B]
            target_lengths      # [B]
        )
        
        return loss