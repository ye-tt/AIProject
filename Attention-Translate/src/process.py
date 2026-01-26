import pandas as pd
import config
from sklearn.model_selection import train_test_split
from tokenizer import ChineseTokenizer, EnglishTokenizer


def process():
    print("开始处理数据")
    # 也可以指定names参数来命名列  df = pd.read_csv(names=["eng", "chn"])
    df = pd.read_csv(
        config.RAW_DATA_PATH,
        encoding="utf-8",
        header=None,
        sep="\t",
        usecols=[0, 1],
    ).dropna()
    df.columns = ["en", "zh"]
    # print(df.head(5))
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    ChineseTokenizer.build_vocab(
        sentences=train_df["zh"].tolist(), vocab_path=config.MODEL_DIR / "zh_vocab.txt"
    )
    EnglishTokenizer.build_vocab(
        sentences=train_df["en"].tolist(), vocab_path=config.MODEL_DIR / "en_vocab.txt"
    )
    # 构建训练集
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODEL_DIR / "zh_vocab.txt")
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODEL_DIR / "en_vocab.txt")
    train_df["zh"] = train_df["zh"].apply(
        lambda x: zh_tokenizer.encode(x, add_sos_eos=False)
    )
    train_df["en"] = train_df["en"].apply(
        lambda x: en_tokenizer.encode(x, add_sos_eos=True)
    )
    # 保存训练集
    train_df.to_json(
        config.PROCESSED_DATA_DIR / "train.jsonl", orient="records", lines=True
    )
    # 构建测试集
    test_df["zh"] = test_df["zh"].apply(
        lambda x: zh_tokenizer.encode(x, add_sos_eos=False)
    )
    test_df["en"] = test_df["en"].apply(
        lambda x: en_tokenizer.encode(x, add_sos_eos=True)
    )
    # 保存测试集
    test_df.to_json(
        config.PROCESSED_DATA_DIR / "test.jsonl", orient="records", lines=True
    )
    print("处理数据结束")


if __name__ == "__main__":
    process()
