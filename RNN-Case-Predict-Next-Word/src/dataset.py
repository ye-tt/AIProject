# 自定义dataset
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import config


class RNNInputDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_json(path, orient="records", lines=True).to_dict(
            orient="records"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_set = self.data[index]
        input_tensor = torch.tensor(data_set["input"], dtype=torch.long)
        target_tensor = torch.tensor(data_set["target"], dtype=torch.long)
        return input_tensor, target_tensor


# 提供一个获取dataLoader 的方法
def get_dataLoader(train=True):
    path = config.PROCESSED_DATA_DIR / ("train.jsonl" if train else "test.jsonl")
    dataSet = RNNInputDataset(path)
    return DataLoader(dataSet, batch_size=config.BATCH_SIZE, shuffle=True)


if __name__ == "__main__":
    train_dataLoader = get_dataLoader(train=True)
    test_dataLoader = get_dataLoader(train=False)

    for input_tensor, target_tendor in train_dataLoader:
        print(input_tensor.shape)  # [batch_size, sel_len]
        print(target_tendor)  # [batch_size]
        break
