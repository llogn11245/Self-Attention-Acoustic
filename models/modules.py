import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class ScaledDotProductAttention(nn.Module):
    ''' 
        Scaled Dot-Product Attention 
    '''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T]
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
    def get_pe(self, seq_len: int) -> torch.Tensor:
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, self.d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        return pe

    def forward(self, x):
        # x is of shape (batch, seq_len, d_model)
        seq_len = x.size(1)
        pe = self.get_pe(seq_len).to(x.device)
        x = x + pe
        return x
    
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class BaseAttention(nn.Module):
    ''' Base module for attentions '''

    def __init__(self, temperature, num_head):
        super().__init__()
        self.temperature = temperature
        self.num_head = num_head
        self.softmax = nn.Softmax(dim=-1)
        self.reset_mem()

    def reset_mem(self):
        # Reset mask
        self.mask = None
        self.k_len = None

    def set_mem(self, prev_att):
        pass

    def compute_mask(self, k, k_len):
        # Make the mask for padded states
        self.k_len = k_len
        bs, ts, _ = k.shape
        self.mask = np.zeros((bs, self.num_head, ts))
        for idx, sl in enumerate(k_len):
            self.mask[idx, :, sl:] = 1  # ToDo: more elegant way?
        self.mask = torch.from_numpy(self.mask).to(
            k_len.device, dtype=torch.bool).view(-1, ts)  # BNxT

    def _attend(self, energy, value):
        attn = energy / self.temperature
        attn = attn.masked_fill(self.mask, -np.inf)
        attn = self.softmax(attn)  # BNxT
        output = torch.bmm(attn.unsqueeze(1), value).squeeze(
            1)  # BNxT x BNxTxD-> BNxD
        return output, attn

class LocationAwareAttention(BaseAttention):
    ''' Location-Awared Attention '''

    def __init__(self, kernel_size, kernel_num, dim, num_head, temperature):
        super().__init__(temperature, num_head)
        self.prev_att = None
        self.loc_conv = nn.Conv1d(
            num_head, kernel_num, kernel_size=2*kernel_size+1, padding=kernel_size, bias=False)
        self.loc_proj = nn.Linear(kernel_num, dim, bias=False)
        self.gen_energy = nn.Linear(dim, 1)
        self.dim = dim

    def reset_mem(self):
        super().reset_mem()
        self.prev_att = None

    def set_mem(self, prev_att):
        self.prev_att = prev_att

    def forward(self, q, k, v):
        bs_nh, ts, _ = k.shape
        bs = bs_nh//self.num_head

        # Uniformly init prev_att
        if self.prev_att is None:
            self.prev_att = torch.zeros((bs, self.num_head, ts)).to(k.device)
            for idx, sl in enumerate(self.k_len):
                self.prev_att[idx, :, :sl] = 1.0/sl

        # Calculate location context
        loc_context = torch.tanh(self.loc_proj(self.loc_conv(
            self.prev_att).transpose(1, 2)))  # BxNxT->BxTxD
        loc_context = loc_context.unsqueeze(1).repeat(
            1, self.num_head, 1, 1).view(-1, ts, self.dim)   # BxNxTxD -> BNxTxD
        q = q.unsqueeze(1)  # BNx1xD

        # Compute energy and context
        energy = self.gen_energy(torch.tanh(
            k+q+loc_context)).squeeze(2)  # BNxTxD -> BNxT
        output, attn = self._attend(energy, v)
        attn = attn.view(bs, self.num_head, ts)  # BNxT -> BxNxT
        self.prev_att = attn

        return output, attn


class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class ResidualConnection(nn.Module):
    
        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)

        def forward(self, x, sublayer):
            return self.norm(x + self.dropout(sublayer(x)))

class ResidualConnectionBase(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, residual):
        return self.norm(x + self.dropout(residual))