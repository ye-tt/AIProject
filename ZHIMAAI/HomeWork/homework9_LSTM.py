from logging import config
import torch
from torch import nn
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from torch import optim


class Config:
    MAX_SEQ_LEN = 128
    SEQ_LEN = 10
    EMBEDED_DIM = 32
    HIDDEN_DIM = 64
    BATCH_SIZE = 16
    LEARN_RATE = 0.001
    EPOCHS = 20
    NUK_TOKEN = "<nuk>"
    PAD_TOKEN = "<pad>"
    PAD_IDEX = 0
    SOS_TOKEN = "<sos>"
    ESO_TOKEN = "<eos>"
    device = "cuda" if torch.cuda.is_available() else "cpu"


def reprocess():
    dataset = load_dataset("shibing624/chinese_text_correction", split="train")
    # print("dataset[0]:",dataset[0])
    split_dataset = dataset.train_test_split(test_size=0.2)
    train_data = split_dataset["train"]
    test_data = split_dataset["test"]
    train_x = train_data["source"]
    train_y = train_data["target"]
    test_x = train_data["source"]
    test_y = train_data["target"]

    return train_x, train_y, test_x, test_y


def build_vocab(sentences):
    # 类方法构建词表并保存
    vocab_set = set()
    # 进度条 tqdm
    for sentence in tqdm(sentences, desc="构建词表"):
        vocab_set.update(list(sentence))

    # vocab_list = [cls.pad_token, cls.unk_token, cls.sos_token, cls.eos_token] + list(vocab_set)
    # 分词会把空格切成一个单独的token,所以要去除空格
    vocab_list = [
        Config.PAD_TOKEN,
        Config.NUK_TOKEN,
        Config.SOS_TOKEN,
        Config.ESO_TOKEN,
    ] + [token for token in vocab_set if token.strip() != ""]
    print(f"词表大小：{len(vocab_list)}")
    word2index = {word: index for index, word in enumerate(vocab_list)}
    index2word = {index: word for index, word in enumerate(vocab_list)}

    return vocab_list, len(vocab_list), word2index, index2word


def sentence_to_ids(text, vocab):
    # 将句子转换为 ID 列表
    return [
        vocab.index(token) if token in vocab else vocab.index(Config.NUK_TOKEN)
        for token in text
    ]


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


def encode(text, word2index, unk_token_index, add_sos_eos=False):
    # 将文本进行分词，然后转化为id
    tokens = list(text)
    if add_sos_eos:
        tokens = [Config.SOS_TOKEN] + tokens + [Config.ESO_TOKEN]
    return [word2index.get(token, unk_token_index) for token in tokens]


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, padding_idx):
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=Config.EMBEDED_DIM,
            padding_idx=Config.PAD_IDEX,
        )
        #
        self.lstm = nn.LSTM(
            input_size=Config.EMBEDED_DIM,
            hidden_size=Config.HIDDEN_DIM,
            batch_first=True,
        )

    def forward(self, x):
        # x.shape [bacth_size,seq_len]
        embeded = self.embedding(x)
        # embeded.shape [bacth_size,seq_len,embed_dim]
        output, _ = self.lstm(embeded)
        # output.shape [bacth_size,seq_len,hidden_size]


def train_one_epoch():
    pass


def train(train_loader, vocab_size):
    print("开始训练...")
    model = LSTMModel(vocab_size, Config.PAD_IDEX).to(Config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), Config.LEARN_RATE)
    for epoch in Config.EPOCHS:
        for inputs, targets in train_loader:
            print("inputs:", inputs)
            print("targets:", targets)
            encoder_inputs, targets = inputs.to(Config.device), targets.to(
                Config.device
            )
            break
            # train_one_epoch()
            # decoder_input=targets[]


if __name__ == "__main__":
    print("开始预处理数据...")
    train_x, train_y, test_x, test_y = reprocess()
    vocab_list, vocab_size, word2index, index2word = build_vocab(train_x)
    train_x_ids = [torch.tensor(sentence_to_ids(sent, vocab_list)) for sent in train_x]
    train_y_ids = [torch.tensor(sentence_to_ids(sent, vocab_list)) for sent in train_y]
    test_x_ids = [torch.tensor(sentence_to_ids(sent, vocab_list)) for sent in test_x]
    test_y_ids = [torch.tensor(sentence_to_ids(sent, vocab_list)) for sent in test_y]
    print("加载train_loader...")
    train_data_set = TensorDataset(train_x, train_y)
    train_loader = DataLoader(
        train_data_set,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    test_data_set = TensorDataset(train_x, train_y)
    test_loader = DataLoader(
        test_data_set, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    train(train_loader, vocab_size)
