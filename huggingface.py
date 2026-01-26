from transformers import AutoModel, AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
from datasets import load_dataset

model = AutoModel.from_pretrained("google-bert/bert-base-chinese")
model = AutoModelForSequenceClassification.from_pretrained(
    "google-bert/bert-base-chinese"
)
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")
tokens = tokenizer.tokenize("我爱自然语言处理")
ids = tokenizer.convert_tokens_to_ids(tokens)
ids2 = [2769, 4263, 5632, 4197, 6427, 6241, 1905, 4415]

tokens = tokenizer.convert_ids_to_tokens(ids2)
print(tokens)

ids3 = tokenizer.encode("我爱自然语言处理")
print(ids)

ids4 = [101, 2769, 4263, 5632, 4197, 6427, 6241, 1905, 4415, 102]


text = "我爱自然语言处理"
input = tokenizer(text)
"""
输出：
{
  'input_ids': [101, 2769, 4263, 5632, 4197, 6427, 6241, 1905, 4415, 102], 
  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
"""
inputs = tokenizer(
    text, padding=True, truncation=True, max_length=128, return_tensors="pt"
)
texts = ["我爱自然语言处理", "我爱人工智能", "我们一起学习"]
encoded = tokenizer(
    texts, padding="max_length", truncation=True, max_length=10, return_tensors="pt"
)

with torch.no_grad():
    outputs = model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        token_type_ids=encoded["token_type_ids"],
    )

dataset_dict = load_dataset("csv", data_files="./data/dataset.csv")
print(dataset_dict)
# DatasetDict({
#     train: Dataset(...)
# })
dataset = dataset_dict["train"]
dataset = dataset.remove_columns(["cat"])
dataset = dataset.filter(
    lambda x: x["review"] is not None
    and x["review"].strip() != ""
    and x["label"] in [0, 1]
)
dataset_dict = dataset.train_test_split(test_size=0.2)

train_dataset = dataset_dict["train"]
test_dataset = dataset_dict["test"]


def tokenize(example):
    encoded = tokenizer(
        example["review"], padding="max_length", truncation=True, max_length=128
    )
    example["input_ids"] = encoded["input_ids"]
    example["attention_mask"] = encoded["attention_mask"]

    return example


train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

dataset_dict.save_to_disk("./data/processed")

from datasets import load_from_disk

dataset_dict = load_from_disk("./data/processed")
train_dataset.set_format(
    type="torch",  # 指定输出为PyTorch张量
    columns=["input_ids", "attention_mask", "label"],  # 需要转换的字段
)

from torch.utils.data import DataLoader

# 训练集DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)


for batch in train_dataloader:
    print(batch)
    break


## {'input_ids': tensor([[...]]),
# 'token_type_ids': tensor([[...]]),
# 'attention_mask':tensor([[...]])}
