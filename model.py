import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class ElectroFormer(nn.Module):
    def __init__(self, num_leads=12, d_model=256, nhead=8, num_layers=6, dim_feedforward=512, dropout=0.1):
        super(ElectroFormer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.positional_encoding = PositionalEncoding(d_model)
        self.embedding = nn.Linear(num_leads, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, num_leads)

    def forward(self, src, src_mask=None):
        batch_size, num_timesteps, num_leads = src.shape
        src = src.reshape(batch_size * num_timesteps, num_leads)
        src = self.embedding(src)
        src = src.reshape(batch_size, num_timesteps, self.d_model)
        src = src.permute(1, 0, 2)  # Convert shape to (num_timesteps, batch_size, d_model)
        src = src * math.sqrt(self.d_model)
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        output = output.permute(1, 0, 2)  # Convert shape back to (batch_size, num_timesteps, num_leads)
        return output

    def get_tgt_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask