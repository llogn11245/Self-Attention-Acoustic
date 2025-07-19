import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder import build_decoder
from .encoder import build_encoder

class AcousticModel(nn.Module):
    def __init__(self, config):
        super(AcousticModel, self).__init__()
        self.encoder = build_encoder(config)
        self.decoder = build_decoder(config)

    def forward(self, inputs, input_lengths, decoder_input, target_lengths, encoder_mask=None):
        encoder_outputs, attn = self.encoder(inputs, encoder_mask)
        decoder_outputs = self.decoder(decoder_input, encoder_outputs, encoder_mask)  
        
        return decoder_outputs
