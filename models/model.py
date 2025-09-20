import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .decoder import build_decoder
from .encoder import build_encoder
from utils.dataset import SpecAugment

class AcousticModel(nn.Module):
    def __init__(self, config):
        super(AcousticModel, self).__init__()
        self.encoder = build_encoder(config)
        self.decoder = build_decoder(config)

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
        encoder_outputs, ctc_out = self.encoder(inputs, encoder_mask)
        decoder_outputs = self.decoder(decoder_input, encoder_outputs, encoder_mask, tfr)  

        return encoder_outputs, ctc_out, decoder_outputs
    
    def recognize(self, enc_inputs, speech_length, target_length=100, enc_mask=None):
        """
        Greedy decode for inference
        Args:
            enc_inputs: [1, time, feature] - batch_size = 1
            speech_length: [1] - lengths of input sequences
            target_length: int - max target length
            enc_mask: [1, time] - mask for encoder inputs
        Returns:
            list of lists: token IDs for each batch item
        """
        encoder_outputs, _ = self.encoder(enc_inputs, enc_mask)
        device = enc_inputs.device
        
        # Khởi tạo decoder input với SOS token
        decoder_input = torch.tensor([[self.sos_id]], device=device)  # [1, 1]
        token_list = []
        
        for step in range(500):
            # Gọi decoder với tfr=0.0 (no teacher forcing)
            with torch.no_grad():
                logits = self.decoder(decoder_input, encoder_outputs, enc_mask, tfr=0.0)
            
            predicted_token = logits[:, -1, :].argmax(dim=-1).item()  
            token_list.append(predicted_token)

            if predicted_token == self.eos_id:
                break
            
            new_token = torch.tensor([[predicted_token]], device=device)  # [1, 1]
            decoder_input = torch.cat([decoder_input, new_token], dim=1)  # [1, step+2]
        
        return [token_list]