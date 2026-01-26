import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


# 构建词表
def build_vocab(origial_corpus):
    words = []
    for sentence in origial_corpus:
        sentence = sentence.lower().replace(".", "")
        words.extend(sentence.split())
    vocab = sorted(set(words))
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    vocab_size = len(vocab)
    print(f"词汇表大小: {vocab_size},词表{vocab}")
    return word_to_idx, idx_to_word, vocab_size


# 生成 CBOW 训练数据
def create_corpus(origial_corpus, word_to_idx, window_size):
    X_context = []
    y_center = []
    context = []
    for doc in origial_corpus:
        words = doc.lower().replace(".", "").split()
        for i in range(window_size, len(words) - window_size):
            y_center.append(word_to_idx[words[i]])
            ctx = [
                words[j]
                for j in range(i - window_size, i + window_size + 1)
                if j != i and 0 <= j < len(words)
            ]
            if ctx:
                X_context.append(ctx)
            # print('X_context',X_context)
            # print('y_center',words[i])
    return X_context, y_center


def preprocess(X_context, word_to_idx):
    X_train = []
    for context in X_context:
        ctx_indices = [word_to_idx[w] for w in context]
        X_train.append(ctx_indices)
    print("X_train", X_train)
    return torch.tensor(X_train, dtype=torch.long)


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        # 词向量查找表
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        avg_embeds = torch.mean(embeds, dim=1)
        return self.fc(avg_embeds)


window_size = 2
embedding_dim = 100
epochs = 10
lr = 0.9
batch_size = 4
origial_corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "Natural language processing is fascinating.",
    "Python makes text analysis easy and fun.",
    "Machine learning models need data to learn.",
    "This is a tiny corpus for testing purposes.",
]
print("\n原始语料：", origial_corpus)
print("构建词表...")
word_to_idx, idx_to_word, vocab_size = build_vocab(origial_corpus)
print("word_to_idx:", word_to_idx)
print("生成 CBOW 训练数据...")
X_context, y_center = create_corpus(origial_corpus, word_to_idx, window_size)
# print("X_context",X_context)
# print("y_center",y_center)
X_train = preprocess(X_context, word_to_idx)
y_train = torch.tensor(y_center, dtype=torch.long)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CBOW(vocab_size, embedding_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
print("\n开始训练...")
for epoch in range(epochs):
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        # 前向传播
        pre_y = model(batch_X)
        # 计算损失
        loss = criterion(pre_y, batch_y)
        # 反向传播计算
        loss.backward()
        # 更新参数
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")


vector_matrix = model.embedding.weight.detach().numpy().copy()
print("词向量矩阵：", vector_matrix)
print("保存模型。。")
torch.save(model.state_dict(), "cbow_model.pth")


test_context = ["the", "brown", "fox", "over"]
try:
    test_indices = [word_to_idx[word] for word in test_context]
except KeyError as e:
    print(f"词 {e} 不在词汇表中")
    exit()

# 加载模型
model_loaded = CBOW(vocab_size, embedding_dim)
model_loaded.load_state_dict(torch.load("cbow_model.pth"))
model_loaded.eval()

# 预测
with torch.no_grad():
    context_tensor = torch.tensor([test_indices], dtype=torch.long)  # [1, 4]
    logits = model_loaded(context_tensor)
    predicted_idx = torch.argmax(logits, dim=1).item()
    predicted_word = idx_to_word[predicted_idx]

print(f"上下文: {test_context}")
print(f"中间词: '{predicted_word}'")
