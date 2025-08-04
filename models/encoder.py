from .atten import MultiHeadAttention, MultiHeadAttentionBlock
from .modules import PositionalEncoding, PositionwiseFeedForward, ResidualConnectionBase
import torch
import torch.nn as nn

class AcousticEncoder(nn.Module):
    def __init__(self, n_head, d_model, d_hidden, dropout=0.1):
        super(AcousticEncoder, self).__init__()
        self.mha = MultiHeadAttention(n_head, d_model, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_hidden, dropout)
        self.pos_enc = PositionalEncoding(d_model)
        self.linear = nn.Linear(d_model, d_hidden)

        self.residual = nn.ModuleList(
            ResidualConnectionBase(d_model, dropout) for _ in range(2)  
        )

    def forward(self, x, mask=None):
        x = self.pos_enc(x)
        atten_out, _ = self.mha(x, x, x, mask)

        x = self.residual[0](x, atten_out)
        x = self.residual[1](x, self.ffn(x))

        x = self.linear(x)
        return x 
    
class InterleaveHybridAcousticEncoder(nn.Module):
    def __init__(self, n_head, d_model, d_hidden, dropout= 0.1):
        super(InterleaveHybridAcousticEncoder, self).__init__()
        self.mha = MultiHeadAttention(n_head, d_model, dropout)
        self.midlayer = ResidualConnectionBase(d_model, dropout)
        self.resi = ResidualConnectionBase(d_model, dropout)
        self.lstm = nn.LSTM(d_model, d_hidden, batch_first= True)
        self.linear = nn.Linear(d_hidden, d_model)
        self.linear2 = nn.Linear(d_model, d_hidden)
    def forward(self, x, mask= None): 
        atten_out, _ = self.mha(x, x, x, mask)

        midlayer = self.midlayer(atten_out, x)

        out, _ = self.lstm(midlayer)
        out = self.linear(out)

        out = self.resi(out, midlayer)
        # out = self.linear2(out)

        return out

def build_encoder(config):
    try: 
        n_head = config['enc']['n_head']
        d_model = config['enc']['d_model']
        d_hidden = config['enc']['d_hidden']
        dropout = config['enc']['dropout']
        type = config['enc']['type']
        if type == 'basic':
            return AcousticEncoder(n_head, d_model, d_hidden, dropout)
        elif type == 'interleave_hybrid': 
            return InterleaveHybridAcousticEncoder(n_head, d_model, d_hidden, dropout)
    except KeyError as e:
        raise ValueError(f"Missing configuration parameter: {e}")