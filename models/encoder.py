from .atten import MultiHeadAttention, MultiHeadAttentionBlock
from .modules import PositionalEncoding, PositionwiseFeedForward, ResidualConnectionBase
import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_hidden, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(n_head, d_model, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_hidden, dropout)
        self.residual = nn.ModuleList(
            ResidualConnectionBase(d_model, dropout) for _ in range(2)  
        )
        
    def forward(self, x, mask=None):
        atten_out, _ = self.mha(x, x, x, mask)
        x = self.residual[0](x, atten_out)
        x = self.residual[1](x, self.ffn(x))
        return x

class AcousticEncoder(nn.Module):
    def __init__(self, n_head, d_model, d_hidden, dropout=0.1, n_layer=1):
        super(AcousticEncoder, self).__init__()
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(n_head, d_model, d_hidden, dropout) for _ in range(n_layer)
        ])
        self.linear = nn.Linear(d_model, d_hidden)

    def forward(self, x, mask=None):
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.linear(x)
        return x 
    
class HybridEncoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_hidden, dropout=0.1):
        super(HybridEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(n_head, d_model, dropout)
        self.midlayer = ResidualConnectionBase(d_model, dropout)
        self.lstm = nn.LSTM(d_model, d_hidden, batch_first=True)
        self.linear = nn.Linear(d_hidden, d_model)
        
    def forward(self, x, mask=None):
        atten_out, _ = self.mha(x, x, x, mask)
        midlayer = self.midlayer(atten_out, x)
        out, _ = self.lstm(midlayer)
        out = self.linear(out)
        return out
    
class InterleaveHybridAcousticEncoder(nn.Module):
    def __init__(self, n_head, d_model, d_hidden, dropout=0.1, n_layer=1):
        super(InterleaveHybridAcousticEncoder, self).__init__()
        self.layers = nn.ModuleList([
            HybridEncoderLayer(n_head, d_model, d_hidden, dropout) for _ in range(n_layer)
        ])
        self.linear2 = nn.Linear(d_model, d_hidden)
        
    def forward(self, x, mask=None): 
        for layer in self.layers:
            x = layer(x, mask)
        out = self.linear2(x)
        return out

def build_encoder(config):
    try: 
        n_head = config['enc']['n_head']
        d_model = config['enc']['d_model']
        d_hidden = config['enc']['d_hidden']
        dropout = config['enc']['dropout']
        type = config['enc']['type']
        n_layer = config['enc']['n_layer']
        
        if type == 'basic':
            return AcousticEncoder(n_head, d_model, d_hidden, dropout, n_layer)
        elif type == 'interleave_hybrid': 
            return InterleaveHybridAcousticEncoder(n_head, d_model, d_hidden, dropout, n_layer)
    except KeyError as e:
        raise ValueError(f"Missing configuration parameter: {e}")