from .atten import MultiHeadAttention, MultiHeadAttentionBlock
from .modules import PositionalEncoding, PositionwiseFeedForward, ResidualConnection, ResidualConnectionBase
import torch
import torch.nn as nn

class AcousticEncoder(nn.Module):
    def __init__(self, n_head, d_model, d_hidden, dropout=0.1):
        super(AcousticEncoder, self).__init__()
        self.mha = MultiHeadAttentionBlock(d_model, n_head, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_hidden, dropout)
        self.pos_enc = PositionalEncoding(d_model)
        self.linear = nn.Linear(d_model, d_hidden)

        self.residual = nn.ModuleList(
            ResidualConnectionBase(d_model, dropout) for _ in range(2)  
        )

    def forward(self, x, mask=None):
        x = self.pos_enc(x)  
        x = self.mha(x, x, x, mask)

        x = self.residual[0](x, self.mha)
        x = self.residual[1](x, self.ffn)

        x = self.linear(x)
        return x 
    
def build_encoder(config):
    try: 
        n_head = config['enc']['n_head']
        d_model = config['enc']['d_model']
        # d_k = config['enc']['d_k']
        # d_v = config['enc']['d_v']
        d_hidden = config['enc']['d_hidden']
        dropout = config['enc']['dropout']

        return AcousticEncoder(n_head, d_model, d_hidden, dropout)
    except KeyError as e:
        raise ValueError(f"Missing configuration parameter: {e}")