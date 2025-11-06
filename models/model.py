import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .decoder import build_decoder
from .encoder import build_encoder
from utils.dataset import SpecAugment

class AcousticModel(nn.Module):
    def __init__(self, config, vocab_size):
        super(AcousticModel, self).__init__()
        self.encoder = build_encoder(config, vocab_size)
        self.decoder = build_decoder(config, vocab_size)

        self.spec_augment = SpecAugment(
            spec_augment=config.get("spec_augment", True),
            mF=config.get("mF", 2),
            F=config.get("F", 30),
            mT=config.get("mT", 2),
            pS=config.get("pS", 0.05),
        )

        self.sos_id = config['sos_id']
        self.eos_id = config['eos_id']
        self.blank_id = config['blank_id']

    def forward(self, inputs, input_lengths, decoder_input, target_lengths, encoder_mask=None, train=True, tfr=1.0):
        if train:
            inputs = inputs.transpose(1, 2)  # (B, T, F) -> (B, F, T)
            inputs = self.spec_augment(inputs, input_lengths)  
            inputs = inputs.transpose(1, 2) # (B, F, T) -> (B, T, F)
        encoder_outputs, encoder_mask, ctc_out = self.encode(inputs, encoder_mask)
        decoder_outputs = self.decode(decoder_input, encoder_outputs, encoder_mask, tfr)  

        return encoder_outputs, ctc_out, decoder_outputs
            
    def encode(self, src, enc_mask):
        encoder_outputs, ctc_out = self.encoder(src, enc_mask)
        return encoder_outputs, enc_mask, ctc_out

    def decode(self, input, enc_out, enc_mask, tfr):
        decoder_outputs = self.decoder(input, enc_out, enc_mask, tfr)
        return decoder_outputs
    
    @torch.no_grad()
    def recognize(self, enc_inputs, enc_mask=None, max_len=100):
        """
        Greedy batch decoding cho inference
        Args:
            enc_inputs: [B, T, F] - batch input
            input_lengths: [B] - chiều dài thực tế mỗi mẫu
            enc_mask: [B, T] - mask cho encoder
            max_len: int - độ dài tối đa của chuỗi decode
        Returns:
            token_lists: list[list[int]] - kết quả token mỗi sample
        """
        device = enc_inputs.device
        B = enc_inputs.size(0)
    
        # Encode
        encoder_outputs, _ = self.encoder(enc_inputs, enc_mask)
    
        # Khởi tạo input decoder: [B, 1] toàn SOS
        decoder_input = torch.full(
            (B, 1), self.sos_id, dtype=torch.long, device=device
        )
    
        # Mask theo batch
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        token_lists = [[] for _ in range(B)]
    
        for _ in range(max_len):
            # Decode
            logits = self.decoder(decoder_input, encoder_outputs, enc_mask, tfr=0.0)
            next_logits = logits[:, -1, :]  # [B, vocab]
            next_tokens = torch.argmax(next_logits, dim=-1)  # [B]
    
            # Gán blank cho những chuỗi đã hoàn thành
            next_tokens = next_tokens.masked_fill(finished, self.blank_id)
    
            # Cập nhật finished
            finished |= (next_tokens == self.eos_id)
    
            # Append token vào danh sách
            for i in range(B):
                token = next_tokens[i].item()
                if token not in [self.sos_id, self.blank_id, self.eos_id]:
                    token_lists[i].append(token)
    
            # Thêm token vào input decoder
            next_tokens = next_tokens.unsqueeze(1)  # [B, 1]
            decoder_input = torch.cat([decoder_input, next_tokens], dim=1)
    
            if finished.all():
                break
    
        return token_lists