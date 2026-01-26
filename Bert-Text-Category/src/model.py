from torch import nn
import config
import torch
from transformers import AutoModel


class MYBertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            config.PRE_TRAINED_DIR / "bert-base-chinese"
        )
        # 二分类，输出为标量
        self.liner = nn.Linear(in_features=self.bert.config.hidden_size, out_features=1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # x.shape [batch_size, seq_len]
        output = self.bert(input_ids, attention_mask, token_type_ids)
        # last_hidden_state shape[batch_size, seql_len,hidden_size]
        last_hidden_state = output.last_hidden_state
        cls_hidden_state = last_hidden_state[:, 0, :]
        # output shape (batch_size)
        output = self.liner(cls_hidden_state).squeeze(-1)

        # 用BCEWithLogitsLoss, output 的形状需要和target 一样
        # output 要不要变形取决于 loss 函数
        output = output.squeeze(-1)  # shape (batch_size)
        # 或者output = output.squeeze(-1)
        return output
