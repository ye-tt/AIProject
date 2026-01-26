import torch.nn as nn
import config
import torch


class MySeq2SeqEncoder(nn.Module):
    # 编码器包含 embeding 和 LSTM
    def __init__(self, vocab_size, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.EMBEDED_DIM,
            padding_idx=padding_idx,
        )
        self.lstm = nn.LSTM(
            input_size=config.EMBEDED_DIM,
            hidden_size=config.HIDDEN_DIM,
            batch_first=True,
        )

    def forward(self, x):
        # x.shape [batch_zise, seq_len]
        embed = self.embedding(x)
        # embed.shape [batch_zise, seq_len, embed_dim]
        output, (hidden, cell) = self.lstm(embed)
        # output.shape [batch_size, seq_len, hidden_size]
        # hidden.shape [1, batch_size, hidden_size]  # num_layers * num_directions, batch, hidden_size
        lengths = (x != self.embedding.padding_idx).sum(
            dim=1
        )  # 计算每个序列的实际长度（不包括填充部分）
        last_hidden_state = output[
            torch.arange(output.shape[0]), lengths - 1
        ]  # 等同于output[torch.arange(output.shape[0]), lengths - 1, :]
        # last_hidden_state.shape [batch_size, hidden_size]
        return last_hidden_state


class MySeq2SeqDecoder(nn.Module):
    # 解码器包含 embeding, LSTM 和 Liner
    def __init__(self, vocab_size, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.EMBEDED_DIM,
            padding_idx=padding_idx,
        )
        self.lstm = nn.LSTM(
            input_size=config.EMBEDED_DIM,
            hidden_size=config.HIDDEN_DIM,
            batch_first=True,
        )
        self.liner = nn.Linear(in_features=config.HIDDEN_DIM, out_features=vocab_size)

    # 一个时间步的forward
    def forward(self, x, hidden_state):
        # x.shape [batch_zise, seq_len] seq_len=1
        # hidden.shape [1, batch_size, hidden_size]  # num_layers * num_directions, batch, hidden_size

        embed = self.embedding(x)
        # embed.shape [batch_zise, seq_len, embed_dim]
        output, hidden_n = self.lstm(embed, hidden_state)
        # output.shape [batch_size, 1, hidden_size]
        output = self.liner(output)
        # output.shape [batch_size, 1, vocab_size]
        return output, hidden_n


class CorrectionModel(nn.Module):
    def __init__(self, vocab_size,  padding_idx):
        super().__init__()
        self.encoder = MySeq2SeqEncoder(
            vocab_size=vocab_size, padding_idx=padding_idx
        )
        self.decoder = MySeq2SeqDecoder(
            vocab_size=vocab_size, padding_idx=padding_idx
        )
    # def forward(self, encoder_input, decoder_input, decoder_hidden=None):
    #     print("LSTMSeq2SeqModel forward")
    #     # encoder_input: [batch_size, src_seq_len]
    #     # decoder_input: [batch_size, trg_seq_len]
        
    #     # 编码
    #     context_vector = self.encoder(encoder_input)
    #     # context_vector: [batch_size, hidden_size]
        
    #     # 解码
    #     if decoder_hidden is None:
    #         decoder_hidden = (
    #             context_vector.unsqueeze(0),
    #             torch.zeros_like(context_vector.unsqueeze(0))
    #         )
        
    #     decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
    #     return decoder_output, decoder_hidden