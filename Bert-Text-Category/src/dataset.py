# 自定义dataset
from torch.utils.data import Dataset, DataLoader
import config
from datasets import load_from_disk


# 提供一个获取dataLoader 的方法
def get_dataLoader(train=True):
    path = str(config.PROCESSED_DATA_DIR / ("train" if train else "test"))
    dateset = load_from_disk(path)
    dateset.set_format(type="torch")
    return DataLoader(dateset, batch_size=config.BATCH_SIZE, shuffle=True)


if __name__ == "__main__":
    train_dataLoader = get_dataLoader(train=True)
    test_dataLoader = get_dataLoader(train=False)
    print(len(train_dataLoader))
    print(len(test_dataLoader))
    for batch in train_dataLoader:
        for k, v in batch.items():
            print(k, "->", v.shape)
        break
