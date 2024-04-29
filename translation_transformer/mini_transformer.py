import torch 
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model 
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        embedding = self.embedding(x) * math.sqrt(self.d_model)
        return embedding


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] * torch.sin(position * div_term)
        pe[:, 1::2] * torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)
        
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, epsilon: float = 10**-6):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1)) # MUltiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Addition
        
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim = True)
        
        return self.alpha * (x - mean) / (std * self.eps) + self.bias


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(torch.relu(x))
        x = self.linear2(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nh, dropout) -> None:
        super().__init__()
        self.d_model = d_model
        self.nh = nh
        assert d_model % nh == 0, 'd_model is not divisble by nh'
        self.d_k = d_model // nh

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
            
        attention_scores = attention_scores.softmax(dim = -1)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores
            
        
    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.nh, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.nh, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.nh, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        x =  x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.nh * self.d_k)
        
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout
        self.norm = LayerNormalization()
        
    def forward(self, x, sublayer):        
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward: FeedForwardLayer, dropout: float):
        super().__init()
        self.self_attention_block = self_attention_block
        self.feed_forward_layer = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self, x, source_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, source_mask))
        x = self.residual_connections[1](x, self.feed_forward_layer)
        
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
            
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward: FeedForwardLayer, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        
    def forward(self, x, encoder_output, source_mask, target_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, target_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, source_mask))
        x = self.residual_connections[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, encoder_output, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)
            
            return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.project = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        return torch.log_softmax(self.project(x), dim=-1)


# Transformer class

class Transformer(nn.Module):

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        source_embed: InputEmbedding,
        target_embed: InputEmbedding,
        source_pos: PositionalEncoding,
        target_pos: PositionalEncoding,
        projection: ProjectionLayer,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embed = source_embed
        self.target_embed = target_embed
        self.source_pos = source_pos
        self.target_pos = target_pos
        self.projection = projection

    def encode(self, source, source_mask):
        source = self.source_embed(source)
        source = self.source_pos(source)

        return self.encoder(source, source_mask)

    def decode(self, encoder_output, source_mask, target, target_mask):
        target = self.target_embed(target)
        target = self.target_pos(target)

        return self.decoder(target, encoder_output, source_mask, target_mask)

    def project(self, x):
        return self.projection(x)


def build_transformer(source_vocab_size, target_vocab_size, source_seq_len, target_seq_len, d_model, nh, n_blocks, d_ff, dropout):
    source_embed = InputEmbedding(d_model, source_vocab_size)
    target_embed = InputEmbedding(d_model, target_vocab_size)

    source_pos = PositionalEncoding(d_model, source_seq_len, dropout)
    target_pos = PositionalEncoding(d_model,  target_seq_len, dropout)

    encoder_blocks = []
    for _ in range(n_blocks):
        encoder_self_attention_block = MultiHeadAttention(d_model, nh, dropout)
        feed_forward_block = FeedForwardLayer(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(n_blocks):
        decoder_cross_attention_block = MultiHeadAttention(d_model, nh, dropout)
        decoder_self_attention_block = MultiHeadAttention(d_model, nh, dropout)
        feed_forward_block = FeedForwardLayer(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            decoder_self_attention_block, decoder_cross_attention_block,
            feed_forward_block,
            dropout,
        )

        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    projection_layer = ProjectionLayer(d_model, target_vocab_size)
    
    transformer = Transformer(encoder, decoder, source_embed, target_embed, source_pos, target_pos, projection_layer)
    
    for x in transformer.parameters():
        if x.dim() > 1:
            nn.init.xavier_uniform_(x)
    
    return transformer
