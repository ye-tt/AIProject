import pandas as pd
import config
from sklearn.model_selection import train_test_split
from tokenizer import JiebaTokenizer


def process():
    print("开始处理数据")
    # 加载数据集
    df = (
        pd.read_csv(config.RAW_DATA_PATH, usecols=["label", "review"], encoding="utf-8")
        .dropna()
        .sample(frac=0.1)
    )
    # df = pd.read_csv(config.RAW_DATA_PATH, encoding="utf-8").dropna()
    # data_set = df[["label", "review"]]
    print(df.head(5))

    # 划分数据集，训练集测试集中正负比例一样,stratify 分层抽样
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"])

    # 构建词表
    JiebaTokenizer.build_vocab(
        train_df["review"].tolist(), config.MODEL_DIR / "vocab.txt"
    )
    # 计算序列的最大长度或者95%的句子最大长度
    # train_df["review"].apply(lambda x: len(tokenizer.encode(x))).max()
    # train_df["review"].apply(lambda x: len(tokenizer.encode(x))).quantile(0.95)

    # 构建tokenizer
    tokenizer = JiebaTokenizer.from_vocab(config.MODEL_DIR / "vocab.txt")

    # 构建训练集并保存
    # apply对 DataFrame 的行或列应用自定义函数，tokenizer.encode() 切词转成id
    train_df["review"] = train_df["review"].apply(
        lambda x: tokenizer.encode(x, config.SEQ_LEN)
    )

    train_df.to_json(
        config.PROCESSED_DATA_DIR / "train.jsonl", orient="records", lines=True
    )

    # 构建测试集并保存
    test_df["review"] = test_df["review"].apply(
        lambda x: tokenizer.encode(x, config.SEQ_LEN)
    )
    test_df.to_json(
        config.PROCESSED_DATA_DIR / "test.jsonl", orient="records", lines=True
    )

    # 每个句子长度不一样，需要填充
    print("处理数据完成")


if __name__ == "__main__":
    process()
