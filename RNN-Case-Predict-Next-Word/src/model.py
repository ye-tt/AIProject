from torch import nn
import config
import torch


class MYRNNModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # emdedding 初始化 -- 随机初始化
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=config.EMBEDDING_SIZE
        )
        # emdedding 初始化 -- 加载预训练完成的
        # embedding_matrix = torch.zeros(vocab_size, config.EMBEDDING_SIZE)
        # self.emdedding = nn.Embedding().from_pretrained()

        # 一般hidden_size>input_size
        self.rnn = nn.RNN(
            input_size=config.EMBEDDING_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            batch_first=True,
        )
        self.liner = nn.Linear(in_features=config.HIDDEN_SIZE, out_features=vocab_size)

    def forward(self, x):
        # x.shape [batch_size, seq_len]
        embed = self.embedding(x)
        # embed.shape [batch_size, seq_len, embedding_dim]
        # rnn 参数： input, h0, h0 不给默认0初始化
        # input.shape (batch_size, seq_len, Hin)  这里 Hin=embedding_dim
        # h0.shape (batch_size,HIDDEN_SIZE)

        # hn 一般用于自己创建RNN hn 作为下一步的h0
        output, hn = self.rnn(embed)
        # output.shape(batch_size,seq_len,HIDDEN_SIZE)
        # h_n.shape (1, batch, HIDDEN_SIZE)
        # output[-1] == h_n.squeeze(0) shape 都是(batch_size,HIDDEN_SIZE)
        # h_n.squeeze(0) 的意思是：移除张量 h_n 的第 0 个维度（如果该维度的大小为 1）
        # output 通常是 所有时间步的输出堆叠而成的张量，output[-1] 最后一个时间步（第 T 步）的输出

        # 最后一个token对应的隐藏状态, last_hideen_size.shape(batch_size,HIDDEN_SIZE)
        last_hideen_size = output[:, -1, :]
        output = self.liner(last_hideen_size)  # shape (batch_size,vocab_size)
        return output
