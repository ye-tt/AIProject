import torch
from datasets import load_dataset
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import config


def save_dataset():
    pass

def load_data_set():
    dataset = load_dataset("shibing624/chinese_text_correction", split="train")
    split_dataset = dataset.train_test_split(test_size=0.2)
    train_data = split_dataset["train"]
    test_data = split_dataset["test"]
    train_df = pd.DataFrame({
        "source": train_data["source"],
        "target": train_data["target"]
    })
    
    test_df = pd.DataFrame({
        "source": test_data["source"],
        "target": test_data["target"]
    })
    # 保存到 CSV 文件
    train_df.to_csv(config.RAW_DATA_DIR / "raw_train_data.csv", index=False, encoding="utf-8")
    test_df.to_csv(config.RAW_DATA_DIR / "raw_test_data.csv", index=False, encoding="utf-8")
    
    print(f"训练集大小: {len(train_df)}")
    print(f"测试集大小: {len(test_df)}")
    print("数据已保存到 CSV 文件")


def build_vocab(sentences):
    # 类方法构建词表并保存
    vocab_set = set()
    # 进度条 tqdm
    for sentence in tqdm(sentences, desc="构建词表"):
        vocab_set.update(list(sentence))

    # vocab_list = [cls.pad_token, cls.unk_token, cls.sos_token, cls.eos_token] + list(vocab_set)
    # 分词会把空格切成一个单独的token,所以要去除空格
    vocab_list = [
        config.PAD_TOKEN,
        config.NUK_TOKEN,
        config.SOS_TOKEN,
        config.ESO_TOKEN,
    ] + [token for token in vocab_set if token.strip() != ""]
    print(f"词表大小：{len(vocab_list)}")
    word2index = {word: index for index, word in enumerate(vocab_list)}
    index2word = {index: word for index, word in enumerate(vocab_list)}

    # 保存词表
    with open(config.VOCAB_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(vocab_list))
    return vocab_list, len(vocab_list), word2index, index2word


def sentence_to_ids(text, word2index):
    # 将句子转换为 ID 列表
    return [
        word2index.get(token,word2index[config.NUK_TOKEN]) for token in text
    ]

def reprocess():
    train_df = pd.read_csv(
        config.RAW_DATA_DIR / "raw_train_data.csv",
        encoding="utf-8",
    ).dropna()
    test_df = pd.read_csv(
        config.RAW_DATA_DIR / "raw_test_data.csv",
        encoding="utf-8",
    ).dropna()

    #构建词表
    all_sentences = train_df["source"].tolist() + train_df["target"].tolist()
    print(all_sentences[0:1])
    print(all_sentences[19999:20000])
    vocab_list, vocab_size, word2index, index2word=build_vocab(all_sentences)

    # 构建训练集
    print("构建训练集...")
    sos_idx = word2index[config.SOS_TOKEN]
    eos_idx = word2index[config.ESO_TOKEN]
    train_df["source"] = train_df["source"].apply(
        lambda x: sentence_to_ids(x, word2index))
    train_df["target"] = train_df["target"].apply(
        lambda x: [sos_idx] + sentence_to_ids(x, word2index) + [eos_idx])
    

    # 保存训练集
    print("保存训练集...")
    train_df.to_json(
        config.PROCESSED_DATA_DIR / "train.jsonl", orient="records", lines=True
    )
    # 构建测试集
    print("构建测试集...")
    test_df["source"] = test_df["source"].apply(
        lambda x: sentence_to_ids(x, word2index))
    test_df["target"] = test_df["target"].apply(
        lambda x: sentence_to_ids(x, word2index))
    print("保存测试集...")
    # 保存测试集
    test_df.to_json(
        config.PROCESSED_DATA_DIR / "test.jsonl", orient="records", lines=True
    )
    
    print("数据预处理完成。")

if __name__ == "__main__":
    print("开始预处理数据...")
    reprocess()