from datasets import load_dataset
from pathlib import Path
from torch import nn
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import re

# ==========================================
# 1. 超参数设置 (Hyperparameters)
# ==========================================
batch_size = 8  # 每次训练样本数
block_size = 64  # 上下文长度（唐诗通常较短，64足够）
max_iters = 3  # 训练迭代次数
eval_interval = 5  # 每隔多少次打印一次效果
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 64  # 向量维度
n_head = 2  # 多头注意力的头数
n_layer = 2  # Transformer 层数
dropout = 0.1


def get_data():
    pass


def load_txt_file(file_path):
    """
    加载本地文本文件，按行分割
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 去除每行的换行符并过滤空行
    lines = [line.strip() for line in lines if line.strip()]
    # 返回字典格式，模拟datasets的结构
    return lines


def construct_gpt_training_data(file_path, device='gpu', block_size=64):
    """
    构造GPT模型的训练数据：x 为诗句的第一个字，y 为整句话
    """
    # 加载文本数据
    data_set = load_txt_file(file_path)

    print(f"加载了 {len(data_set)} 行诗歌数据")
    for i, poem in enumerate(data_set[:3]):  # 显示前3行
        print(f"诗歌 {i + 1}: {poem[:50]}{'...' if len(poem) > 50 else ''}")

    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")

    # 如果tokenizer没有pad_token，则设置为unk_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    # 计算词汇表大小
    vocab_size = tokenizer.vocab_size
    print(f"\n=== 词汇表信息 ===")
    print(f"词汇表大小 (vocab_size): {vocab_size}")

    # 按句子分割诗歌，构造 (第一个字, 整句话) 对
    print(f"\n开始按句子分割诗歌...")
    sentence_pairs = []  # 存储 (首字, 整句) 对

    for poem in data_set:
        # 移除标题
        poem2 = ""
        if ':' in poem:
            poem2 = poem.split(':', 1)[1]

        # 按 '。' 和 '，' 分割句子
        sentences = re.split(r'[。，]', poem2)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 0:
                first_char = sentence[0]  # 第一个字
                sentence_pairs.append((first_char, sentence))

    print(f"提取了 {len(sentence_pairs)} 个诗句对")
    for i, (first_char, sentence) in enumerate(sentence_pairs[:5]):
        print(f"  诗句 {i + 1}: 首字='{first_char}', 整句='{sentence}'")

    # 对每对进行tokenize
    x_list = []  # 首字的token id
    y_list = []  # 整句的token ids

    for first_char, sentence in sentence_pairs:
        # tokenize 首字
        first_char_tokens = tokenizer.encode(first_char, add_special_tokens=False)
        if len(first_char_tokens) > 0:
            x_list.append(first_char_tokens[0])  # 取第一个token
        else:
            x_list.append(tokenizer.unk_token_id)  # 如果失败用unk token

        # tokenize 整句话
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        y_list.append(sentence_tokens)

    print(f"成功tokenize {len(x_list)} 个训练样本")

    # 转换为tensor
    # x: (样本数,) - 每个首字的token id
    x = torch.tensor(x_list, dtype=torch.long)

    # y: 需要padding到相同长度 (样本数, max_句子长度)
    max_sentence_length = min(max(len(tokens) for tokens in y_list), block_size)

    y_padded = []
    for tokens in y_list:
        padded = tokens + [tokenizer.pad_token_id] * (max_sentence_length - len(tokens))
        y_padded.append(padded)
    y = torch.tensor(y_padded, dtype=torch.long)

    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print(f"最大句子长度: {max_sentence_length}")

    # 移动到指定设备
    x, y = x.to(device), y.to(device)

    print(f"最终 x shape: {x.shape}")
    print(f"最终 y shape: {y.shape}")
    print(f"x device: {x.device}")
    print(f"y device: {y.device}")

    # 显示前几个样本
    for i in range(min(3, len(x))):
        first_char_decoded = tokenizer.decode([x[i].item()])
        sentence_decoded = tokenizer.decode(y[i].tolist(), skip_special_tokens=True)
        print(f"样本 {i + 1}: 首字='{first_char_decoded}', 整句='{sentence_decoded}'")

    return x, y, vocab_size, tokenizer


def decode_tokens(tokenizer, tokens):
    """
    解码token为文本
    """
    return tokenizer.decode(tokens, skip_special_tokens=True)


class Head(nn.Module):
    """ 单个注意力头 """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        # 计算注意力得分 (Affinity)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # 因果掩码
        wei = nn.functional.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # 加权求和
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """ 多头注意力 """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """ 简单的线性层 + 激活函数 """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer 块 """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # 残差连接
        x = x + self.ffwd(self.ln2(x))  # 残差连接
        return x


class PoetryGPT(nn.Module):
    """ 完整的 GPT 模型 """

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # 训练时：idx: (batch_size, 1) 首字, targets: (batch_size, max_sentence_length)
        # 生成时：idx: (batch_size, seq_len) 部分序列
        B = idx.shape[0]

        if targets is not None:
            # 训练模式：将首字和目标句子拼接
            # idx: (B, 1), targets: (B, T_t)
            # 移除targets最后一个token避免越界
            T_t = targets.shape[1]
            x_input = torch.cat([idx, targets[:, :-1]], dim=1)  # (B, 1+T_t-1) = (B, T_t)
            T = x_input.shape[1]
        else:
            # 生成模式：直接使用idx
            x_input = idx
            T = idx.shape[1]

        tok_emb = self.token_embedding_table(x_input)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb  # (B, T, n_embd)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        ## logits(B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            logits_flat = logits.reshape(-1, V)  # (B*T, vocab_size)
            targets_flat = targets.reshape(-1)  # (B*T,)
            loss = nn.functional.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50):
        """
        生成文本，使用top-k采样避免乱码
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # 裁剪上下文
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # 调整温度

            # Top-k 采样，只保留概率最高的k个token
            top_k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
            probs = nn.functional.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    # batch(input_ids,token_type_ids,attention_mask,labels)
    for x, y in tqdm(dataloader, desc="Training"):
        x_batch = x.unsqueeze(1).to(device)  # (batch_size, 1) 首字作为单个token序列
        y_batch = y.to(device)  # (batch_size, max_sentence_length)

        logits, loss = model(x_batch, y_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


if __name__ == '__main__':
    file_path = Path(__file__).resolve().parent / "data" / "poetry.txt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x, y, vocab_size, tokenizer = construct_gpt_training_data(
        file_path=file_path,
        device=device,
    )

    # 随机抽样 50% 的数据
    total_samples = len(x)
    sample_size = total_samples // 2

    # 生成随机索引
    indices = torch.randperm(total_samples)[:sample_size]

    # 按索引抽样
    x = x[indices]
    y = y[indices]

    print(f"抽样后数据大小: x={x.shape}, y={y.shape}")

    # 1. 构建 Dataset
    dataset = TensorDataset(x, y)

    # 2. 构建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练时通常打乱
    )
    model = PoetryGPT().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("开始训练")
    for itr in range(max_iters):
        loss = train_one_epoch(model, dataloader, optimizer, device)
        print(f"loss:{loss}")

    # # 生成测试
    print("\n--- 训练完成，开始作诗 ---")
    # 测试多种起始词
    start_words = input("请输入任意人名，词语：")
    print(f"\n--- 生成 '{start_words}' 的藏头诗 ---")
    model.eval()
    results =[]
    with torch.no_grad():
        for start_word in list(start_words):
            # 编码起始词
            start_encoded = tokenizer.encode(start_word, add_special_tokens=False)
            start_tensor = torch.tensor(start_encoded, dtype=torch.long, device=device).unsqueeze(0)  # 添加批次维度

            # 生成诗歌，使用较低的温度和top-k采样
            generated_tokens = model.generate(start_tensor, max_new_tokens=50, temperature=0.7, top_k=30)
            generated_text = decode_tokens(tokenizer, generated_tokens[0].tolist())

            # 清理生成的文本
            clean_generated = generated_text[1:].replace('[UNK]', '').replace('[PAD]', '').replace('[CLS]', '').replace(
                '[SEP]',
                '')
            results.append(clean_generated)
        print(f"生成的诗句:\n{clean_generated}")


