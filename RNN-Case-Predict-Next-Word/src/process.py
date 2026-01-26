## 对原始数据进行数据预处理并保存词表，训练数据和测试数据到相对文件

import pandas as pd
import config
from sklearn.model_selection import train_test_split
import jieba
from tqdm import tqdm
from tokenizer import JiebaTokenizer


def build_dataSet(sentences, tokenizer):

    # 列表推导式得到 index2word
    # indexed_sentences =[tokenizer.encode(sentence) for sentence in sentences]
    # print(indexed_sentences[0:10])

    indexed_sentences = []
    # #下面代码与上行等效
    for sentence in sentences:
        indexed_sentences.append(tokenizer.encode(sentence))
    print("indexed_sentences:", indexed_sentences[0:10])

    data_set = []
    for sentence in tqdm(indexed_sentences, desc="构建数据集"):
        for i in range(len(sentence) - config.SEQ_LEN):
            input = sentence[i : i + config.SEQ_LEN]
            target = sentence[i + config.SEQ_LEN]
            data_set.append({"input": input, "target": target})
    print("data_set:", data_set[0:3])
    return data_set


def process():
    print("开始处理数据")

    # 读取文件
    raw_data_path = config.RAW_DATA_PATH

    # 按照Json 文件格式选择不同orient
    df = pd.read_json(raw_data_path, lines=True, orient="records").sample(
        frac=0.05, random_state=42
    )
    # print("head:",df.head()[0:10])
    # print(df['dialog'])
    sentences = []
    for dialog in df["dialog"]:
        for sentence in dialog:
            sentences.append(sentence.split("：")[1])

    # print("sentences:",sentences[0:10])

    # 划分数据集
    train_sentences, test_sentences = train_test_split(sentences, test_size=0.2)

    # 构建词表并保存
    JiebaTokenizer.build_vocab(train_sentences, config.MODEL_DIR / "vocab.txt")

    # 构建训练集
    tokenizer = JiebaTokenizer.from_vocab(config.MODEL_DIR / "vocab.txt")
    train_data_set = build_dataSet(train_sentences, tokenizer)
    # 保存训练集
    pd.DataFrame(train_data_set).to_json(
        config.PROCESSED_DATA_DIR / "train.jsonl", orient="records", lines=True
    )

    # #构建测试集
    test_data_set = build_dataSet(test_sentences, tokenizer)
    # #保存测试集
    pd.DataFrame(test_data_set).to_json(
        config.PROCESSED_DATA_DIR / "test.jsonl", orient="records", lines=True
    )

    print("数据处理完成")


if __name__ == "__main__":
    process()
