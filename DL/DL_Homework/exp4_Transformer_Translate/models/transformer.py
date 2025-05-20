import torch
import torch.nn as nn
import math
import copy
from models.modules import MultiHeadedAttention, FeedForward, SublayerConnection, Generator


class Embedding(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, device='cuda'):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0.0, max_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2, device=device) * (-math.log(1e4) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x += self.pe[:, :x.size(1)]
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, h, d_model, d_ff=256, dropout=0.1):
        super(Encoder, self).__init__()
        self.self_attn = MultiHeadedAttention(h, d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, mask))
        return self.norm(self.sublayer2(x, self.feed_forward))


class Decoder(nn.Module):
    def __init__(self, h, d_model, d_ff=256, dropout=0.1):
        super(Decoder, self).__init__()
        self.self_attn = MultiHeadedAttention(h, d_model)
        self.src_attn = MultiHeadedAttention(h, d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer2(x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.norm(self.sublayer3(x, self.feed_forward))


class Transformer(nn.Module):
    def __init__(self, tokenizer, h=8, d_model=256, E_N=2, D_N=2, device='cuda'):
        super(Transformer, self).__init__()
        self.encoder = nn.ModuleList([Encoder(h, d_model) for _ in range(E_N)])
        self.decoder = nn.ModuleList([Decoder(h, d_model) for _ in range(D_N)])
        vocab_size = tokenizer.get_vocab_size()
        self.src_embed = Embedding(d_model, vocab_size)
        self.tgt_embed = Embedding(d_model, vocab_size)
        self.src_pos = PositionalEncoding(d_model, device=device)
        self.tgt_pos = PositionalEncoding(d_model, device=device)
        self.generator = Generator(d_model, vocab_size)

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        for enc_layer in self.encoder:
            src = enc_layer(src, src_mask)
        return src

    def decode(self, memory, tgt, src_mask, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        for dec_layer in self.decoder:
            tgt = dec_layer(tgt, memory, src_mask, tgt_mask)
        return tgt

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), tgt, src_mask, tgt_mask)
