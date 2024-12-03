import torch
import torch.nn as nn
from utils import MultiHeadAttention, FeedForward
from encoder import Embedding, PositionalEncoder, Encoder

class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim=2048, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.encoder_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embedding_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, source_mask, target_mask):
        self_attn_output = self.self_attention(x, x, x, target_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        enc_attn_output = self.encoder_attention(x, encoder_output, encoder_output, source_mask)
        x = self.norm2(x + self.dropout(enc_attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_len, num_heads, num_layers, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.position_embedding = PositionalEncoder(embedding_dim, max_seq_len, dropout)
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, trg, encoder_output, src_mask, trg_mask):
        x = self.embedding(trg)
        x = self.position_embedding(x)
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, source_max_seq_len, target_max_seq_len, embedding_dim, num_heads, num_layers, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(source_vocab_size, embedding_dim, source_max_seq_len, num_heads, num_layers, dropout)
        self.decoder = Decoder(target_vocab_size, embedding_dim, target_max_seq_len, num_heads, num_layers, dropout)
        self.final_linear = nn.Linear(embedding_dim, target_vocab_size)

    def make_source_mask(self, src):
        return (src != 0).unsqueeze(1).unsqueeze(2)

    def make_target_mask(self, trg):
        trg_pad_mask = (trg != 0).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool()
        return trg_pad_mask & trg_sub_mask

    def forward(self, src, trg):
        src_mask = self.make_source_mask(src)
        trg_mask = self.make_target_mask(trg)
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(trg, enc_output, src_mask, trg_mask)
        output = self.final_linear(dec_output)
        return output