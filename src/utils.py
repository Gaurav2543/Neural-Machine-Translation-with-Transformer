import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def create_bpe_tokenizer(texts, vocab_size):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"], vocab_size=vocab_size)
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer

def read_data(source_file, target_file):
    source_data = open(source_file, encoding='utf-8').read().strip().split("\n")
    target_data = open(target_file, encoding='utf-8').read().strip().split("\n")
    return source_data, target_data

class TranslateDataset(torch.utils.data.Dataset):
    def __init__(self, source_tokenizer, target_tokenizer, source_data, target_data, source_max_seq_len, target_max_seq_len):
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_data = source_data
        self.target_data = target_data
        self.source_max_seq_len = source_max_seq_len
        self.target_max_seq_len = target_max_seq_len

    def __len__(self):
        return len(self.source_data)

    def __getitem__(self, idx):
        source_encoded = self.source_tokenizer.encode(self.source_data[idx])
        target_encoded = self.target_tokenizer.encode(self.target_data[idx])
        
        source_ids = [self.source_tokenizer.token_to_id("[SOS]")] + source_encoded.ids + [self.source_tokenizer.token_to_id("[EOS]")]
        target_ids = [self.target_tokenizer.token_to_id("[SOS]")] + target_encoded.ids + [self.target_tokenizer.token_to_id("[EOS]")]
        
        source_ids = source_ids[:self.source_max_seq_len]
        target_ids = target_ids[:self.target_max_seq_len]
        
        source_ids += [self.source_tokenizer.token_to_id("[PAD]")] * (self.source_max_seq_len - len(source_ids))
        target_ids += [self.target_tokenizer.token_to_id("[PAD]")] * (self.target_max_seq_len - len(target_ids))
        
        return {
            "source_ids": torch.tensor(source_ids),
            "target_ids": torch.tensor(target_ids),
        }

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)
        self.final_projection = nn.Linear(embedding_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        Q = self.query_projection(query)
        K = self.key_projection(key)
        V = self.value_projection(value)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(query.device)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim=-1)
        
        x = torch.matmul(self.dropout(attention), V)
        
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.embedding_dim)
        x = self.final_projection(x)
        
        return x

class FeedForward(nn.Module):
    def __init__(self, embedding_dim, ff_dim=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x