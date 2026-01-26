from turtle import forward
import torch.nn as nn
import config
import torch


class Attention(nn.Module):

    # 一个时间步的注意力机制
    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden.shape [batch_size, 1, hidden_size]
        # encoder_outputs.shape [batch_size, seq_len, hidden_size]

        # 计算注意力分数
        # torch.bmm((batch_size, 1, hidden_size), (batch_size, hidden_size, seq_len)) ->(batch_size, 1, seq_len)
        # torch.bmm(decoder_hidden, encoder_outputs.transpose(1, 2))
        attention_scores = torch.bmm(decoder_hidden, encoder_outputs.transpose(1, 2))

        # 计算注意力权重
        attn_weights = torch.softmax(attention_scores, dim=-1)
        # attn_weights.shape [batch_size,1, seq_len]

        # 计算上下文向量[batch_size, 1, hidden_size]
        return torch.bmm(attn_weights, encoder_outputs)


class TranslationEncoder(nn.Module):
    # 编码器包含 embeding 和 RNN
    def __init__(self, vocab_size, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            padding_idx=padding_idx,
        )
        self.gru = nn.GRU(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            batch_first=True,
        )

    # 解码器 需要用到编码器每一步的输出，所以还要返回output
    def forward(self, x):
        # x.shape [batch_zise, seq_len]
        embed = self.embedding(x)
        # embed.shape [batch_zise, seq_len, embed_dim]
        output, _ = self.gru(embed)
        # output.shape [batch_size, seq_len, hidden_size]
        # hidden.shape [1, batch_size, hidden_size]  # num_layers * num_directions, batch, hidden_size
        lengths = (x != self.embedding.padding_idx).sum(
            dim=1
        )  # 计算每个序列的实际长度（不包括填充部分）
        last_hidden_state = output[
            torch.arange(output.shape[0]), lengths - 1
        ]  # 等同于output[torch.arange(output.shape[0]), lengths - 1, :]
        # last_hidden_state.shape [batch_size, hidden_size]
        return output, last_hidden_state


class TranslationDecoder(nn.Module):
    # 解码器包含 embeding, RNN 和 Liner
    def __init__(self, vocab_size, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            padding_idx=padding_idx,
        )
        self.gru = nn.GRU(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            batch_first=True,
        )
        self.attention = Attention()
        self.liner = nn.Linear(
            in_features=2 * config.HIDDEN_SIZE, out_features=vocab_size
        )

    # 一个时间步的forward
    def forward(self, x, hidden_0, encoder_outputs):
        # x.shape [batch_zise, seq_len] seq_len=1
        # hidden.shape [1, batch_size, hidden_size]  # num_layers * num_directions, batch, hidden_size

        embed = self.embedding(x)
        # embed.shape [batch_zise, seq_len, embed_dim]
        output, hidden_n = self.gru(embed, hidden_0)
        # output.shape [batch_size, 1, hidden_size]

        # 应用注意力机制(output, encoder_outputs)
        context_vector = self.attention(output, encoder_outputs)
        # context_vector.shape [batch_size, 1, hidden_size]

        # 融合信息,将context_vector 和decoder 的 output 拼接
        combined = torch.cat([output, context_vector], dim=-1)
        # combined.shape [batch_size, 1, hidden_size * 2]

        output = self.liner(combined)
        # output.shape [batch_size, 1, vocab_size]
        return output, hidden_n


class TranslationModel(nn.Module):
    def __init__(self, zh_vocab_size, en_vocab_size, zh_padding_idx, en_padding_idx):
        super().__init__()
        self.encoder = TranslationEncoder(
            vocab_size=zh_vocab_size, padding_idx=zh_padding_idx
        )
        self.decoder = TranslationDecoder(
            vocab_size=en_vocab_size, padding_idx=en_padding_idx
        )
