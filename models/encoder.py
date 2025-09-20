from .atten import MultiHeadAttention, MultiHeadAttentionBlock
from .modules import PositionalEncoding, PositionwiseFeedForward, ResidualConnectionBase
import torch
import torch.nn as nn
    
class HybridEncoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_hidden, dropout=0.1):
        super(HybridEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(n_head, d_model, dropout)
        self.midlayer = ResidualConnectionBase(d_model, dropout)
        self.lstm = nn.LSTM(d_model, d_hidden, batch_first=True)
        self.linear = nn.Linear(d_hidden, d_model)
        
    def forward(self, x, mask=None):
        atten_out, mask = self.mha(x, x, x, mask)
        midlayer = self.midlayer(atten_out, x)
        out, _ = self.lstm(midlayer)
        out = self.linear(out)
        out = out + midlayer
        return out
    
class InterleaveHybridAcousticEncoder(nn.Module):
    def __init__(self, n_head, d_model, d_hidden, vocab_size, dropout=0.1, n_layer=1):
        super(InterleaveHybridAcousticEncoder, self).__init__()
        self.layers = nn.ModuleList([
            HybridEncoderLayer(n_head, d_model, d_hidden, dropout) for _ in range(n_layer)
        ])
        self.linear2 = nn.Linear(d_model, d_hidden)
        self.ctc_proj = nn.Linear(d_hidden, vocab_size)

    def forward(self, x, mask=None): 
        for layer in self.layers:
            x = layer(x, mask)
        out = self.linear2(x)
        ctc_out = self.ctc_proj(out)  # [B, T, vocab_size]

        return out, ctc_out      # [B, T, d_hidden] & [B, T, vocab_size]

def build_encoder(config, vocab_size):
    try: 
        n_head = config['enc']['n_head']
        d_model = config['enc']['d_model']
        d_hidden = config['enc']['d_hidden']
        dropout = config['enc']['dropout']
        n_layer = config['enc']['n_layer']

        return InterleaveHybridAcousticEncoder(n_head, d_model, d_hidden, vocab_size, dropout, n_layer)
    except KeyError as e:
        raise ValueError(f"Missing configuration parameter: {e}")