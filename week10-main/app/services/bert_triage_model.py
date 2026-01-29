from app.core.config import settings
from transformers import AutoTokenizer, AutoModel
from torch import nn

class BertTriageModel(nn.Module):
    def __init__(self,num_classes:int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(settings.pretrained_model_path/'bert-base-chinese')
        self.dropout = nn.Dropout(0.1)  # 通常BERT后面会加dropout
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask,token_type_ids):
        # shape: [batch_size, seq_len]
        output = self.bert(input_ids, attention_mask, token_type_ids)

        last_hidden_state = output.last_hidden_state
        # last_hidden_state.shape: [batch_size, seq_len, hidden_size]

        cls_hidden_state = last_hidden_state[:,0,:]
        # cls_hidden_state.shape: [batch_size, hidden_size]
        cls_hidden_state = self.dropout(cls_hidden_state)
        output = self.linear(cls_hidden_state).squeeze(-1)
        # output.shape: [batch_size]
        return output

