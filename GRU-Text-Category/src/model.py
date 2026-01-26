from torch import nn
import config
import torch


class MYLSTMModel(nn.Module):
    def __init__(self, vocab_size, papadding_idx):
        super().__init__()
        # emdedding 初始化 -- 随机初始化,padding_idx 指定<pad> 的索引
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.EMBEDDING_SIZE,
            padding_idx=papadding_idx,
        )
        # emdedding 初始化 -- 加载预训练完成的
        # embedding_matrix = torch.zeros(vocab_size, config.EMBEDDING_SIZE)
        # self.emdedding = nn.Embedding().from_pretrained()
        self.papadding_idx = papadding_idx
        # 一般hidden_size>input_size
        self.lstm = nn.LSTM(
            input_size=config.EMBEDDING_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            batch_first=True,
        )
        # 二分类，输出为标量
        self.liner = nn.Linear(in_features=config.HIDDEN_SIZE, out_features=1)

    def forward(self, x: torch.Tensor):
        # x.shape [batch_size, seq_len]
        embed = self.embedding(x)
        # embed.shape [batch_size, seq_len, embedding_dim]
        # rnn 参数： input, h0, h0 不给默认0初始化
        # input.shape (batch_size, seq_len, Hin)  这里 Hin=embedding_dim
        # h0.shape (batch_size,HIDDEN_SIZE)

        # hn 一般用于自己创建RNN hn 作为下一步的h0
        output, hn = self.lstm(embed)
        # output.shape(batch_size,seq_len,HIDDEN_SIZE)

        # 获取每个样本真是的最后一个token对应的隐藏状态, last_hideen.shape (batch_size,HIDDEN_SIZE)
        # 因为序列有填充，所以要取非pad 的最后一个时间步隐藏状态
        batch_indexes = torch.arange(0, output.shape[0])  # 得到 0 到bathsize 的一维列表
        # x != 0 得到bool 矩阵，pad 的地方为0，其他为1. 按照seq_len 方向求和，就是最后一个不为pad 的index+1
        length = (x != self.papadding_idx).sum(dim=1)
        # 通过列表索引得到last_hideen
        last_hideen = output[batch_indexes, length - 1]

        output = self.liner(last_hideen)  # shape (batch_size,1)

        # 用BCEWithLogitsLoss, output 的形状需要和target 一样
        # output 要不要变形取决于 loss 函数
        output = output.squeeze(-1)  # shape (batch_size)
        # 或者output = output.squeeze(-1)
        return output
