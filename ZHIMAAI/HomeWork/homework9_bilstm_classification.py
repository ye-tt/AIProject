import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
from torch import embedding, no_grad
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import SGD, Adam, RMSprop
import torch
from tqdm import tqdm
import matplotlib.pyplot as plot


from sympy import frac


def build_char_list(train_df):
    all_chars = []
    for sentence in train_df["text"]:
        all_chars.extend(list(sentence.strip()))
    # 过滤低频字
    char_counts = Counter(all_chars)
    sorted_chars = sorted([c for c, count in char_counts.items() if count >= 1])
    char_list = ["<unk>"] + list(sorted_chars)
    return char_list


def get_input_tragt_data(raw_df, char2idx, label2idx):
    input_ids = []
    # print("111111111", len(raw_df["text"]))
    # print("22222222222", len(raw_df["label"]))
    max_seq_len = 64
    for text in raw_df["text"]:
        ids = [char2idx.get(char, 0) for char in text]
        if len(ids) < max_seq_len:
            ids = ids + [0] * (64 - len(ids))
        else:
            ids = ids[:max_seq_len]
        input_ids.append(ids)
    # print("333333333333", len(input_ids))
    label_vectors = np.zeros((len(raw_df["text"]), len(label2idx)))
    for i, labels in enumerate(raw_df["label"]):
        if "," in labels:
            for label in labels.split(","):
                idx = label2idx[label]
                label_vectors[i, idx] = 1.0
        else:
            idx = label2idx[labels]
            label_vectors[i, idx] = 1.0
    # input_ids.shape[]

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(
        label_vectors, dtype=torch.float
    )


def reprocess():
    raw_data_path = Path(__file__).parent / "data" / "train.csv"
    df = pd.read_csv(raw_data_path, encoding="utf-8").sample(frac=0.01).dropna()
    train_df, test_df = train_test_split(df, test_size=0.2)
    label_set = set()
    for lables in train_df["label"]:
        if "," in lables:
            label_set.update(lables.split(","))
        else:
            label_set.add(lables)
    # 构建字符表
    char_list = build_char_list(train_df)
    # print(char_list)
    char2idx = {char: index for index, char in enumerate(char_list)}
    idx2char = {index: char for index, char in enumerate(char_list)}
    label2idx = {char: index for index, char in enumerate(label_set)}
    idx2label = {index: char for index, char in enumerate(label_set)}

    vocab_size = len(char_list)
    train_x, train_y = get_input_tragt_data(train_df, char2idx, label2idx)
    print(f"train_x.shape: {train_x.shape}, train_y.shape: {train_y.shape}")  # 新增

    test_x, test_y = get_input_tragt_data(test_df, char2idx, label2idx)
    print(f"train_x.shape: {train_x.shape}, train_y.shape: {train_y.shape}")  # 新增

    num_labels = len(label2idx)
    return train_x, test_x, train_y, test_y, vocab_size, num_labels


class MYLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, out_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, num_layers=2, batch_first=True, dropout=0.1
        )
        self.liner = nn.Linear(hidden_size, out_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x.shape [batch_size, seq_len]
        embedded = self.embedding(x)
        # embedded.shape [batch_size, seq_len,embed_dim]
        out_put, hidden_state = self.lstm(embedded)
        # out_put.shape [batch_size, seq_len,hidden_size]
        # 取最后一个时间步的输出，最后一个token
        last_hidden = out_put[:, -1, :]
        # out_put.shape [batch_size, hidden_size]
        out = self.dropout(last_hidden)
        logits = self.liner(out)
        # logits.shape [batch_size,out_dim]
        return logits


def train(optimizer_name, lr):
    print("开始训练")
    BATCH_SIZE = 64
    EPOCHS = 10
    EMBED_DIM = 128
    HIDDEN_SIZE = 256
    print("预处理")
    train_x, test_x, train_y, test_y, vocab_size, OUT_DIM = reprocess()
    print(f"train_x.shape: {train_x.shape}, train_y.shape: {train_y.shape}")  # 新增
    print(f"OUT_DIM: {OUT_DIM}")
    print("加载数据")
    data_set = TensorDataset(train_x, train_y)
    data_loader = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True)
    model = MYLSTMModel(vocab_size, EMBED_DIM, HIDDEN_SIZE, OUT_DIM)

    criterion = nn.BCEWithLogitsLoss()

    if optimizer_name == "Adam":
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    else:  # RMSprop
        optimizer = RMSprop(model.parameters(), lr=lr)

    model.train()
    loss_history = []

    for epoch in range(EPOCHS):
        print(f"Epoch:{epoch}")
        total_loss = 0.0
        for texts, labels in tqdm(data_loader, desc="Training"):
            optimizer.zero_grad()
            pred_y = model(texts)
            # pred_y shape [batch_size,OUT_DIM]
            # labels shape[batch_size,OUT_DIM]
            loss = criterion(pred_y, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("total_loss:", total_loss)
        avg_loss = total_loss / len(data_loader)
        print("avg_loss:", avg_loss)
        loss_history.append(avg_loss)
    print(loss_history)


def show_result():
    experiments = [("SGD", 0.1), ("Adam", 0.001), ("RMSprop", 0.001)]

    results = {}
    for opt, lr in experiments:
        label = f"{opt} (lr={lr})"
        losses = train(opt, lr)
        results[label] = losses

    for label, losses in results:
        plot.plot(losses, label=label)
    plot.title("Multi-Label Classification Loss")
    plot.xlabel("Epochs")
    plot.ylabel("BCE Loss")
    plot.show()


if __name__ == "__main__":
    show_result
