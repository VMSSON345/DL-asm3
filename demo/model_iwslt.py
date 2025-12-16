import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import Counter

# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. Embedding + Positional Encoding
# ============================================================

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=200):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (2 * (i + 1) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# ============================================================
# 2. Attention
# ============================================================

def attention(q, k, v, mask=None, dropout=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)
    if dropout:
        scores = dropout(scores)

    return torch.matmul(scores, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)

        scores = attention(q, k, v, mask, self.dropout)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.h * self.d_k)

        return self.out(concat)


# ============================================================
# 3. Feed Forward
# ============================================================

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear_2(self.dropout(F.relu(self.linear_1(x))))


# ============================================================
# 4. Layer Norm
# ============================================================

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        return self.alpha * (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + self.eps) + self.bias


# ============================================================
# 5. Encoder / Decoder Layers
# ============================================================

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attention = MultiHeadAttention(heads, d_model, dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = x + self.dropout_1(self.attention(self.norm_1(x), self.norm_1(x), self.norm_1(x), mask))
        x = x + self.dropout_2(self.ff(self.norm_2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        x = x + self.dropout_1(self.attn_1(self.norm_1(x), self.norm_1(x), self.norm_1(x), tgt_mask))
        x = x + self.dropout_2(self.attn_2(self.norm_2(x), enc_out, enc_out, src_mask))
        x = x + self.dropout_3(self.ff(self.norm_3(x)))
        return x


# ============================================================
# 6. Encoder / Decoder
# ============================================================

class Encoder(nn.Module):
    def __init__(self, vocab, d_model, N, heads, dropout):
        super().__init__()
        self.embed = Embedder(vocab, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, dropout) for _ in range(N)])
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.pe(self.embed(src))
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab, d_model, N, heads, dropout):
        super().__init__()
        self.embed = Embedder(vocab, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, dropout) for _ in range(N)])
        self.norm = Norm(d_model)

    def forward(self, tgt, enc_out, src_mask, tgt_mask):
        x = self.pe(self.embed(tgt))
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.norm(x)


# ============================================================
# 7. Transformer
# ============================================================

class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=256, N=4, heads=4, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(tgt_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc = self.encoder(src, src_mask)
        dec = self.decoder(tgt, enc, src_mask, tgt_mask)
        return self.out(dec)


# ============================================================
# 8. Tokenizer (WORD-LEVEL)
# ============================================================

class SimpleTokenizer:
    def __init__(self, vocab_size=30000, min_freq=2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.PAD = "<pad>"
        self.BOS = "<bos>"
        self.EOS = "<eos>"
        self.UNK = "<unk>"
        self.word2id = {}
        self.id2word = {}

    def fit(self, texts):
        counter = Counter()
        for t in texts:
            counter.update(t.lower().split())

        vocab = [self.PAD, self.BOS, self.EOS, self.UNK]
        vocab += [w for w, c in counter.items() if c >= self.min_freq][: self.vocab_size]
        self.word2id = {w: i for i, w in enumerate(vocab)}
        self.id2word = {i: w for w, i in self.word2id.items()}

    def encode(self, text, max_len=100):
        ids = [self.word2id.get(w, self.word2id[self.UNK]) for w in text.lower().split()]
        ids = ids[:max_len]
        return [self.word2id[self.BOS]] + ids + [self.word2id[self.EOS]]

    def decode(self, ids):
        words = []
        for i in ids:
            w = self.id2word.get(int(i), self.UNK)
            if w not in [self.PAD, self.BOS, self.EOS]:
                words.append(w)
        return " ".join(words)

    def vocab_size_(self):
        return len(self.word2id)


# ============================================================
# 9. MASKS + GREEDY DECODE
# ============================================================

def make_src_mask(src):
    return (src != 0).unsqueeze(1).unsqueeze(2)

def make_tgt_mask(tgt):
    T = tgt.size(1)
    pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
    seq_mask = torch.tril(torch.ones((T, T), device=tgt.device)).bool()
    return pad_mask & seq_mask


@torch.no_grad()
def greedy_decode(model, src, src_mask, tgt_tok, max_len=80):
    ys = torch.tensor([[tgt_tok.word2id[tgt_tok.BOS]]], device=device)
    src = src.unsqueeze(0)

    for _ in range(max_len):
        out = model(src, ys, src_mask.unsqueeze(0), make_tgt_mask(ys))
        next_word = out[:, -1].argmax(-1).item()
        ys = torch.cat([ys, torch.tensor([[next_word]], device=device)], dim=1)
        if next_word == tgt_tok.word2id[tgt_tok.EOS]:
            break

    return ys[0].tolist()
