import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, d_k)
        """
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, d_k)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        q = self.W_q(query)  # (batch_size, seq_len, d_model)
        k = self.W_k(key)    # (batch_size, seq_len, d_model)
        v = self.W_v(value)  # (batch_size, seq_len, d_model)
        
        # Split heads
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, d_k)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, d_k)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, d_k)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, v)
        
        # Combine heads
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(context)
        
        return output, attention_weights

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class PositionalEncoding(nn.Module):
    """
    Positional Encoding module
    """
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_seq_length, d_model)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class LanguageEmbedding(nn.Module):
    """
    Language Embedding module for multilingual support
    """
    def __init__(self, num_languages, d_model):
        super(LanguageEmbedding, self).__init__()
        self.language_embeddings = nn.Embedding(num_languages, d_model)
        
    def forward(self, x, language_id):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
            language_id: Tensor, shape [batch_size]
        """
        # Get language embeddings
        lang_emb = self.language_embeddings(language_id).unsqueeze(1)  # (batch_size, 1, d_model)
        
        # Add language embeddings to input
        return x + lang_emb

class EncoderLayer(nn.Module):
    """
    Encoder Layer module
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
            mask: Tensor, shape [batch_size, 1, seq_len, seq_len]
        """
        # Self-attention with residual connection and layer normalization
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x

class DecoderLayer(nn.Module):
    """
    Decoder Layer module
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
            enc_output: Tensor, shape [batch_size, enc_seq_len, d_model]
            look_ahead_mask: Tensor, shape [batch_size, 1, seq_len, seq_len]
            padding_mask: Tensor, shape [batch_size, 1, seq_len, enc_seq_len]
        """
        # Self-attention with residual connection and layer normalization
        attn1_output, _ = self.self_attn(x, x, x, look_ahead_mask)
        x = self.norm1(x + self.dropout1(attn1_output))
        
        # Cross-attention with residual connection and layer normalization
        attn2_output, attention_weights = self.cross_attn(x, enc_output, enc_output, padding_mask)
        x = self.norm2(x + self.dropout2(attn2_output))
        
        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x, attention_weights

class MultilingualEncoder(nn.Module):
    """
    Multilingual Encoder module
    """
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, 
                 num_languages, max_seq_length=5000, dropout=0.1):
        super(MultilingualEncoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.language_embedding = LanguageEmbedding(num_languages, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, src_lang_id, mask=None):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
            src_lang_id: Tensor, shape [batch_size]
            mask: Tensor, shape [batch_size, 1, 1, seq_len]
        """
        seq_len = x.size(1)
        
        # Token embeddings
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # Add language embeddings
        x = self.language_embedding(x, src_lang_id)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Encoder layers
        for layer in self.enc_layers:
            x = layer(x, mask)
            
        return x  # (batch_size, seq_len, d_model)

class MultilingualDecoder(nn.Module):
    """
    Multilingual Decoder module
    """
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, 
                 num_languages, max_seq_length=5000, dropout=0.1):
        super(MultilingualDecoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.language_embedding = LanguageEmbedding(num_languages, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, tgt_lang_id, look_ahead_mask=None, padding_mask=None):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
            enc_output: Tensor, shape [batch_size, enc_seq_len, d_model]
            tgt_lang_id: Tensor, shape [batch_size]
            look_ahead_mask: Tensor, shape [batch_size, 1, seq_len, seq_len]
            padding_mask: Tensor, shape [batch_size, 1, 1, enc_seq_len]
        """
        seq_len = x.size(1)
        attention_weights = {}
        
        # Token embeddings
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # Add language embeddings
        x = self.language_embedding(x, tgt_lang_id)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Decoder layers
        for i, layer in enumerate(self.dec_layers):
            x, attention = layer(x, enc_output, look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer{i+1}'] = attention
            
        return x, attention_weights  # (batch_size, seq_len, d_model), attention_weights

class MultilingualTransformer(nn.Module):
    """
    Multilingual Transformer model for machine translation
    """
    def __init__(self, vocab_size, d_model=512, num_heads=8, d_ff=2048, 
                 num_layers=6, num_languages=4, max_seq_length=5000, dropout=0.1):
        super(MultilingualTransformer, self).__init__()
        
        self.encoder = MultilingualEncoder(
            vocab_size, d_model, num_heads, d_ff, num_layers, 
            num_languages, max_seq_length, dropout
        )
        
        self.decoder = MultilingualDecoder(
            vocab_size, d_model, num_heads, d_ff, num_layers, 
            num_languages, max_seq_length, dropout
        )
        
        self.final_layer = nn.Linear(d_model, vocab_size)
        
        # Language ID mapping
        self.language_map = {
            'en': 0,  # English
            'fr': 1,  # French
            'es': 2,  # Spanish
            'vi': 3   # Vietnamese
        }
        
    def create_masks(self, src, tgt):
        """
        Create masks for transformer
        
        Args:
            src: Tensor, shape [batch_size, src_seq_len]
            tgt: Tensor, shape [batch_size, tgt_seq_len]
        """
        # Padding mask for src (1 for tokens, 0 for padding)
        src_padding_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_seq_len)
        
        # Padding mask for tgt (1 for tokens, 0 for padding)
        tgt_padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, tgt_seq_len)
        
        # Look-ahead mask for decoder (1 for tokens to attend to, 0 for tokens to mask)
        tgt_seq_len = tgt.size(1)
        look_ahead_mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len)), diagonal=1).eq(0)
        look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, tgt_seq_len, tgt_seq_len)
        
        # Combine padding mask with look-ahead mask
        combined_mask = torch.logical_and(tgt_padding_mask, look_ahead_mask)
        
        return src_padding_mask, combined_mask
    
    def get_language_id(self, language):
        """
        Get language ID from language code
        
        Args:
            language: str, language code ('en', 'fr', 'es', 'vi')
        """
        return self.language_map.get(language, 0)  # Default to English if unknown
    
    def forward(self, src, tgt, src_lang, tgt_lang):
        """
        Forward pass
        
        Args:
            src: Tensor, shape [batch_size, src_seq_len]
            tgt: Tensor, shape [batch_size, tgt_seq_len]
            src_lang: str, source language code ('en', 'fr', 'es', 'vi')
            tgt_lang: str, target language code ('en', 'fr', 'es', 'vi')
        """
        # Create masks
        src_padding_mask, combined_mask = self.create_masks(src, tgt)
        
        # Get language IDs
        batch_size = src.size(0)
        src_lang_id = torch.tensor([self.get_language_id(src_lang)] * batch_size, device=src.device)
        tgt_lang_id = torch.tensor([self.get_language_id(tgt_lang)] * batch_size, device=tgt.device)
        
        # Encoder
        enc_output = self.encoder(src, src_lang_id, src_padding_mask)
        
        # Decoder
        dec_output, attention_weights = self.decoder(
            tgt, enc_output, tgt_lang_id, combined_mask, src_padding_mask
        )
        
        # Final linear layer
        logits = self.final_layer(dec_output)
        
        return logits, attention_weights
    
    def translate(self, src, src_lang, tgt_lang, max_length=100):
        """
        Translate a source sequence to target language
        
        Args:
            src: Tensor, shape [batch_size, src_seq_len]
            src_lang: str, source language code ('en', 'fr', 'es', 'vi')
            tgt_lang: str, target language code ('en', 'fr', 'es', 'vi')
            max_length: int, maximum length of generated sequence
        """
        batch_size = src.size(0)
        device = src.device
        
        # Get language IDs
        src_lang_id = torch.tensor([self.get_language_id(src_lang)] * batch_size, device=device)
        tgt_lang_id = torch.tensor([self.get_language_id(tgt_lang)] * batch_size, device=device)
        
        # Create src padding mask
        src_padding_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_seq_len)
        
        # Encode source sequence
        enc_output = self.encoder(src, src_lang_id, src_padding_mask)
        
        # Initialize decoder input with start token
        dec_input = torch.ones((batch_size, 1), dtype=torch.long, device=device) * 2  # <s> token
        
        # Store output tokens and attention weights
        output_tokens = []
        attention_weights_list = []
        
        # Generate target sequence
        for i in range(max_length):
            # Create look-ahead mask
            look_ahead_mask = torch.triu(torch.ones((i+1, i+1)), diagonal=1).eq(0)
            look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(1).to(device)
            
            # Create padding mask for decoder input
            dec_padding_mask = (dec_input != 0).unsqueeze(1).unsqueeze(2)
            combined_mask = torch.logical_and(dec_padding_mask, look_ahead_mask)
            
            # Decode
            dec_output, attention_weights = self.decoder(
                dec_input, enc_output, tgt_lang_id, combined_mask, src_padding_mask
            )
            
            # Get predictions for the next token
            predictions = self.final_layer(dec_output[:, -1:, :])  # (batch_size, 1, vocab_size)
            predicted_id = torch.argmax(predictions, dim=-1)  # (batch_size, 1)
            
            # Store predictions and attention weights
            output_tokens.append(predicted_id)
            attention_weights_list.append(attention_weights)
            
            # Break if end token is predicted
            if (predicted_id == 3).all():  # </s> token
                break
            
            # Concatenate predicted token to decoder input
            dec_input = torch.cat([dec_input, predicted_id], dim=1)
        
        # Concatenate all output tokens
        output_sequence = torch.cat(output_tokens, dim=1)
        
        return output_sequence, attention_weights_list

def create_multilingual_transformer(vocab_size, num_languages=4, d_model=512, num_heads=8, 
                                   d_ff=2048, num_layers=6, max_seq_length=5000, dropout=0.1):
    """
    Create a multilingual transformer model
    
    Args:
        vocab_size: int, size of vocabulary
        num_languages: int, number of languages
        d_model: int, dimension of model
        num_heads: int, number of attention heads
        d_ff: int, dimension of feed-forward network
        num_layers: int, number of encoder/decoder layers
        max_seq_length: int, maximum sequence length
        dropout: float, dropout rate
    """
    model = MultilingualTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        num_languages=num_languages,
        max_seq_length=max_seq_length,
        dropout=dropout
    )
    
    # Initialize parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return model
