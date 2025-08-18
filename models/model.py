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

        self.sos_id = config['sos_id']
        self.eos_id = config['eos_id']

    def forward(self, inputs, input_lengths, decoder_input, target_lengths, encoder_mask=None, train=True):
        if train:
            inputs = inputs.transpose(1, 2)  # (B, T, F) -> (B, F, T)
            inputs = SpecAugment(inputs)
            inputs = inputs.transpose(1, 2) # (B, F, T) -> (B, T, F)
        encoder_outputs = self.encoder(inputs, encoder_mask)
        decoder_outputs = self.decoder(decoder_input, encoder_outputs, encoder_mask)  
        
        return decoder_outputs
    
    def recognize(self, enc_inputs, speech_length, target_length=100, enc_mask=None):
        """
        Greedy decoding for inference
        Args:
            enc_inputs: [batch, time, feature]
            speech_length: [batch] - lengths of input sequences
            target_length: [batch] - lengths of target sequences
            enc_mask: [batch, time] - mask for encoder inputs
        Returns:
            list of lists: token IDs for each batch item
        """
        encoder_outputs = self.encoder(enc_inputs, enc_mask)
        batch_size = enc_inputs.size(0)
        device = enc_inputs.device
        sos_id = self.sos_id
        eos_id = self.eos_id

        h = [torch.zeros(1, self.decoder.hidden_size).to(device) for _ in range(self.decoder.num_layers)]
        c = [torch.zeros(1, self.decoder.hidden_size).to(device) for _ in range(self.decoder.num_layers)]
        context = torch.zeros(1, self.decoder.hidden_size).to(device)

        token_list = []
        current_token = sos_id 

        for _ in range(target_length-1):  # -1 because we don't predict the EOS token
            embedded = self.decoder.embedding(torch.tensor([current_token], device=device))
            rnn_input = torch.cat([embedded, context], dim=1)

            h[0], c[0] = self.decoder.rnn[0](rnn_input, (h[0], c[0]))
            for i in range(1, self.decoder.num_layers):
                h[i], c[i] = self.decoder.rnn[i](h[i-1], (h[i], c[i]))

            query = h[-1].unsqueeze(0).unsqueeze(0)  # [1, 1, 1, hidden_size]
            key = value = encoder_outputs.unsqueeze(0)  # [1, 1, time, hidden_size]
            if enc_mask is not None:
                attn_mask = enc_mask.unsqueeze(0)  # [1, 1, time]
            else:
                attn_mask = None
            new_context, attn = self.decoder.attention(query, key, value, mask=attn_mask)
            new_context = new_context.squeeze(1).squeeze(1)  # [1, hidden_size]

            char_input = torch.cat([h[-1], new_context], dim=1)
            output = self.decoder.mlp(char_input)
            predicted_token = torch.argmax(output, dim=1).item()

            token_list.append(predicted_token)

            if predicted_token == eos_id:
                break

            current_token = predicted_token

        return [token_list]