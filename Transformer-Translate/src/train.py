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
        # 去掉最后一个<eos>作为输入  shape (batch_size, trg_seq_len-1),(batch_size, seq_len)
        decoder_inputs = targets[:, :-1]
        # 去掉第一个<sos>作为目标  shape (batch_size, trg_seq_len-1),(batch_size, seq_len)
        decoder_targets = targets[:, 1:]

        # 前向传播
        src_pad_mask = encoder_inputs == model.zh_embedding.padding_idx
        # trt_len = decoder_inputs.size(1)
        trt_len = decoder_inputs.shape[1]
        tgt_mask = model.transformer.generate_square_subsequent_mask(
            sz=trt_len, device=device
        )
        decoder_outputs = model(encoder_inputs, decoder_inputs, src_pad_mask, tgt_mask)
        # decoder_outputs.shape [batch_zise, sql_len, en_vocab_size]
        # 计算损失
        # CrossEntropyLoss 需要 (N, C) 或 (N, C, d1...)，N = batch_size * seq_len,这里 reshape
        decoder_outputs = decoder_outputs.reshape(-1, decoder_outputs.shape[-1])
        decoder_targets = decoder_targets.reshape(-1)
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
