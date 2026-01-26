import torch
from dataset import get_dataLoader
import config
from tokenizer import ChineseTokenizer, EnglishTokenizer
from model import TranslationModel
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, targets in tqdm(dataloader, desc="训练"):
        encoder_inputs = inputs.to(device)  # shape (batch_size, src_seq_len)
        targets = targets.to(device)  # shape (batch_size, trg_seq_len)
        # 去掉最后一个<eos>作为输入  shape (batch_size, trg_seq_len-1)
        decoder_inputs = targets[:, :-1]
        # 去掉第一个<sos>作为目标  shape (batch_size, trg_seq_len-1)
        decoder_targets = targets[:, 1:]
        # 编码阶段
        encoder_outputs, context_vector = model.encoder(
            encoder_inputs
        )  # shape [batch_size, vocab_size]

        # 解码阶段
        decoder_outputs = []
        # decoder_hidden_0 shape [1, batch_size, hidden_size]
        # unsqueeze在指定位置插入一个大小为 1 的新维度
        decoder_hidden = context_vector.unsqueeze(0)  # 加上第一维，维度为1
        seq_len = decoder_inputs.shape[1]
        for i in range(seq_len):
            # decoder_input = decoder_inputs[:, i].unsqueeze(1)  # decoder_inputs[:, i]会变成(batch_size),unsqueeze 后shape [batch_size, 1]
            decoder_input = decoder_inputs[:, i : i + 1]  # shape [batch_size, 1]
            # decoder forward 参数变化，需要增加encoder_outputs
            decoder_output, decoder_hidden = model.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # decoder_output shape: [batch_size, 1, vocab_size]
            # seq_len 个[batch_size, 1, vocab_size] 的列表
            decoder_outputs.append(decoder_output)

        # decoder_outputs： [tensor([batch_size,1,vocab_size])] ->[batch_size * seq_len, vocab_size]
        # 将多个时间步的输出拼接在一起，形成完整的输出序列
        # shape [batch_size, seq_len, vocab_size]
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        # shape [batch_size * seq_len, vocab_size]
        decoder_outputs = decoder_outputs.reshape(-1, decoder_outputs.shape[-1])

        # decoder_targets: [batch_size, seq_len] -> [batch_size * seq_len]
        decoder_targets = decoder_targets.reshape(-1)

        # 计算损失
        # CrossEntropyLoss 需要 (N, C) 或 (N, C, d1...)，N = batch_size * seq_len,这里 reshape
        loss = loss_fn(decoder_outputs, decoder_targets)

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def train():
    # 确定 device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载数据集
    dataloader = get_dataLoader(train=True)
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODEL_DIR / "en_vocab.txt")
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODEL_DIR / "zh_vocab.txt")
    model = TranslationModel(
        zh_tokenizer.vocab_size,
        en_tokenizer.vocab_size,
        zh_tokenizer.pad_token_index,
        en_tokenizer.pad_token_index,
    ).to(device)
    # CrossEntropyLoss 里已经包含了softmax
    # Specifies a target value that is ignored and does not contribute to the input gradient
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_index)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARN_RATE)

    writer = SummaryWriter(config.LOGS_DIR / time.strftime("%Y-%m-%d_%H-%M-%S"))

    best_loss = float("inf")
    # 开始训练
    for epoch in range(config.EPOCHS):
        print(f"Epoch:{epoch}")
        # 训练一个 epoch 逻辑
        loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        print(f"loss:{loss}")

        # 记录训练结果
        writer.add_scalar("loss", loss, epoch)
        # 保存模型
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), config.MODEL_DIR / "best.pth")  # .pt pytorch
            # 用torch.load()加载
            print("模型保存成功")

    writer.close()


if __name__ == "__main__":
    train()
    # tensor1 = torch.randn(3, 4)
    # print(tensor1)
    # tensor2 = tensor1[:, 3]
    # print("tensor2.shape", tensor2.shape)
    # print(tensor2)
    # tensor3 = tensor1.unsqueeze(1)
    # print("tensor3.shape", tensor3.shape)
    # print(tensor3)
