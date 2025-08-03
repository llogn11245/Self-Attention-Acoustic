import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import ScaledDotProductAttention

class AcousticDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, embed_dropout=0.1, var_dropout=0.2, sos_id=1, eos_id=2, pad_id=0):
        super(AcousticDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.rnn = nn.ModuleList([
            nn.LSTMCell(embedding_dim + hidden_size, hidden_size)
            if i == 0 else 
            nn.LSTMCell(hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        self.attention = ScaledDotProductAttention(temperature=hidden_size**0.5)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, vocab_size)
        )
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.var_dropout = nn.Dropout(var_dropout)

    def forward(self, decoder_input, encoder_outputs, encoder_mask=None):
        """
        Args:
            decoder_input: [batch, max_len] (bắt đầu bằng SOS)
            encoder_outputs: [batch, time, hidden]
            encoder_mask: [batch, time] (mask cho encoder)
        """
        max_len = decoder_input.size(1)
        batch_size = decoder_input.size(0)

        h = [torch.zeros(batch_size, self.hidden_size).to(encoder_outputs.device) 
             for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size).to(encoder_outputs.device) 
             for _ in range(self.num_layers)]
        context = torch.zeros(batch_size, self.hidden_size).to(encoder_outputs.device)

        outputs = []
        embedded = self.embedding(decoder_input)  # [B, max_len, embed]
        embedded = self.embed_dropout(embedded)  # Apply dropout to target tokens
        for t in range(max_len):
            rnn_input = torch.cat([embedded[:, t, :], context], dim=1)
            h[0], c[0] = self.rnn[0](rnn_input, (h[0], c[0]))
            for i in range(1, self.num_layers):
                h[i], c[i] = self.rnn[i](h[i-1], (h[i], c[i]))
                h[i] = self.var_dropout(h[i])  # variational recurrent dropout
            
            query = h[-1].unsqueeze(1).unsqueeze(1)  # [B, 1, 1, hidden]
            key = value = encoder_outputs.unsqueeze(1)  # [batch, 1, time, hidden]
            if encoder_mask is not None:
                attn_mask = encoder_mask.unsqueeze(1)  # [B, 1, time]
            else:
                attn_mask = None
                
            context, attn = self.attention(query, key, value, mask=attn_mask)
            context = context.squeeze(1)  # [B, 1, hidden]
            context = context.squeeze(1)  # [B, hidden]
            char_input = torch.cat([h[-1], context], dim=1)
            output = self.mlp(char_input)
            outputs.append(output)

        logits = torch.stack(outputs, dim=1)  # [batch, max_len, vocab_size]
        return logits  # [B, max_len, vocab]
    
def build_decoder(config):
    try: 
        vocab_size = config['vocab_size']
        embedding_dim = config['dec']['embed_dim']
        hidden_size = config['dec']['d_hidden']
        num_layers = config['dec']['num_layers']
        embed_dropout = config['dec']['embed_dropout']
        var_dropout = config['dec']['var_dropout']
        sos_id = config['sos_id']
        eos_id = config['eos_id']
        pad_id = config['pad_id']

        return AcousticDecoder(vocab_size, embedding_dim, hidden_size, num_layers, embed_dropout, var_dropout, 
                               sos_id, eos_id, pad_id)
    except KeyError as e:
        raise ValueError(f"Missing configuration parameter: {e}")