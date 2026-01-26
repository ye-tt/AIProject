import jieba
import config
from tqdm import tqdm


class JiebaTokenizer:
    unk_token = "<unk>"

    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2index = {word: index for index, word in enumerate(vocab_list)}
        self.index2word = {index: word for index, word in enumerate(vocab_list)}
        self.unk_token_index = self.word2index[self.unk_token]

    @staticmethod
    def tokenize(text):
        # 分词
        return jieba.lcut(text)

    def encode(self, text):
        # 将文本进行分词，然后转化为id
        tokens = self.tokenize(text)
        return [self.word2index.get(token, self.unk_token_index) for token in tokens]

    @classmethod
    def build_vocab(cls, sentences, vocab_path):
        # 类方法构建词表并保存
        vocab_set = set()
        # 进度条 tqdm
        for sentence in tqdm(sentences, desc="构建词表"):
            vocab_set.update(cls.tokenize(sentence))

        vocab_list = [cls.unk_token] + list(vocab_set)
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


if __name__ == "__main__":
    tokenizer = JiebaTokenizer.from_vocab(config.MODEL_DIR / "vocab.txt")
    print(f"词表大小：{tokenizer.vocab_size}")
    print(f"词表大小：{tokenizer.unk_token}")
    token_indexes = tokenizer.encode("今天天气不错")
    print(token_indexes)
