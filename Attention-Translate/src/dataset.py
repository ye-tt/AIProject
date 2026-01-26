from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import config
from torch.nn.utils.rnn import pad_sequence


class Seq2SeqDataSet(Dataset):
    def __init__(self, path):
        self.data = pd.read_json(path, orient="records", lines=True).to_dict(
            orient="records"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_set = self.data[index]
        input_tensor = torch.tensor(data_set["zh"], dtype=torch.long)
        target_tensor = torch.tensor(data_set["en"], dtype=torch.long)
        return input_tensor, target_tensor


def collate_fn(batch):
    # batch 是一个二元组列表，包含多个样本，每个样本是一个 (input_tensor, target_tensor) 元组 [(input_tensor1, target_tensor1),(input_tensor2, target_tensor2)....]
    # input_tensor= batch[0][0], target_tensor=batch[0][1]
    # input_tensor= [item[0] for item in batch], target_tensor=[item[1] for item in batch]
    input_tensors, target_tensors = zip(*batch)

    # 使用 pad_sequence 对输入和目标进行填充
    # input_tensors_padded.shape: [batch_size, max_seq_len]
    input_tensors_padded = pad_sequence(
        input_tensors, batch_first=True, padding_value=0
    )
    target_tensors_padded = pad_sequence(
        target_tensors, batch_first=True, padding_value=0
    )

    return input_tensors_padded, target_tensors_padded


def get_dataLoader(train=True):
    path = config.PROCESSED_DATA_DIR / ("train.jsonl" if train else "test.jsonl")
    dataSet = Seq2SeqDataSet(path)
    # collate_fn: 是一个用于自定义如何将多个样本（samples）合并成一个 batch 的函数
    # 默认情况下，DataLoader 使用 torch.utils.data.default_collate 来自动把一批样本“堆叠”（stack）成张量,但前提是：所有样本的形状必须一致
    return DataLoader(
        dataSet, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )


if __name__ == "__main__":
    train_dataLoader = get_dataLoader(train=True)
    test_dataLoader = get_dataLoader(train=False)
    for input_tensor, target_tendor in train_dataLoader:
        print(input_tensor.shape)  # [batch_size, seq_len]
        print(target_tendor.shape)  # [batch_size, seq_len]
        break
