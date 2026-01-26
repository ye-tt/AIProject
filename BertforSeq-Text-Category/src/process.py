from fileinput import filename
from tkinter.font import names
import config
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer


def process():
    print("开始处理数据")
    dataset = load_dataset(path="csv", data_files=str(config.RAW_DATA_PATH))["train"]
    dataset = dataset.remove_columns(["cat"])
    dataset = dataset.filter(lambda x: x["review"] is not None)
    dataset = dataset.cast_column("label", ClassLabel(names=["negative", "positive"]))
    # print(dataset.features["label"].int2str(0)) ClassLabel 内有int2str，str2int

    # 划分数据集
    # stratify_by_column 分层抽样
    dataset_dict = dataset.train_test_split(test_size=0.2, stratify_by_column="label")
    # 创建Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR / "bert-base-chinese")

    # 构建数据集
    def batch_encode(batch):
        inputs = tokenizer(
            batch["review"],
            padding="max_length",
            truncation=True,
            max_length=config.MAX_SEQ_LEN,
        )
        inputs["labels"] = batch["label"]
        return inputs

    dataset_dict = dataset_dict.map(
        batch_encode, batched=True, remove_columns=["review", "label"]
    )
    print(dataset_dict)
    # 保存数据集
    dataset_dict.save_to_disk(config.PROCESSED_DATA_DIR)

    print("处理数据完成")


if __name__ == "__main__":
    process()
