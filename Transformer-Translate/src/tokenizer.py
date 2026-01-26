import jieba
import config
from tqdm import tqdm
from nltk import TreebankWordTokenizer, TreebankWordDetokenizer


class BaseTokenizer:
    unk_token = "<unk>"
    pad_token = "<pad>"
    sos_token = "<sos>"
    eos_token = "<eos>"

    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2index = {word: index for index, word in enumerate(vocab_list)}
        self.index2word = {index: word for index, word in enumerate(vocab_list)}
        self.unk_token_index = self.word2index[self.unk_token]
        self.pad_token_index = self.word2index[self.pad_token]
        self.sos_token_index = self.word2index[self.sos_token]
        self.eos_token_index = self.word2index[self.eos_token]

    @classmethod
    # 申明返回类型-> list[str]
    def tokenize(cls, text) -> list[str]:
        # 分词
        return jieba.lcut(text)

    def encode(self, text, add_sos_eos=False):
        # 将文本进行分词，然后转化为id
        tokens = self.tokenize(text)
        if add_sos_eos:
            tokens = [self.sos_token] + tokens + [self.eos_token]
        return [self.word2index.get(token, self.unk_token_index) for token in tokens]

    @classmethod
    def build_vocab(cls, sentences, vocab_path):
        # 类方法构建词表并保存
        vocab_set = set()
        # 进度条 tqdm
        for sentence in tqdm(sentences, desc="构建词表"):
            vocab_set.update(cls.tokenize(sentence))

        # vocab_list = [cls.pad_token, cls.unk_token, cls.sos_token, cls.eos_token] + list(vocab_set)
        # 分词会把空格切成一个单独的token,所以要去除空格
        vocab_list = [cls.pad_token, cls.unk_token, cls.sos_token, cls.eos_token] + [
            token for token in vocab_set if token.strip() != ""
        ]
        print(f"词表大小：{len(vocab_list)}")

        # 保存词表
        with open(vocab_path, "w", encoding="utf-8") as f:
            f.write("\n".join(vocab_list))

    @classmethod
    def from_vocab(cls, vocab_path):
        # 类方法从指定路径得到个vocab_list，然后构造JiebaTokenizer 对象并返回
        # 读取词表
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_list = [line.strip() for line in f.readlines()]
        return cls(vocab_list)


class ChineseTokenizer(BaseTokenizer):
    @classmethod
    def tokenize(cls, text) -> list[str]:
        # 分词
        return list(text)


class EnglishTokenizer(BaseTokenizer):
    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()

    @classmethod
    def tokenize(cls, text) -> list[str]:
        # 分词
        return cls.tokenizer.tokenize(text)

    def decode(self, indexes):
        tokens = [self.index2word[index] for index in indexes]
        return self.detokenizer.detokenize(tokens)


if __name__ == "__main__":
    # tokenizer = BaseTokenizer.from_vocab(config.MODEL_DIR / "vocab.txt")
    # print(f"词表大小：{tokenizer.vocab_size}")
    # print(f"词表大小：{tokenizer.unk_token}")
    # token_indexes = tokenizer.encode("今天天气不错", 128)
    # print(token_indexes)

    # tokenizer = TreebankWordTokenizer()
    # word_tokenize = tokenizer.tokenize(
    #     "On a $50,000 mortgage of 30 years at 8 percent, the monthly payment would be $366.88."
    # )
    # print(word_tokenize)
    pass
