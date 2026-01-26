import torch
from dataset import get_dataLoader
from model import MYLSTMModel
import config
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
from tokenizer import JiebaTokenizer


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs = inputs.to(device)  # shape (batch_size, seq_len)
        targets = targets.to(device)  # shape (batch_size)

        # 前向传播
        outputs = model(inputs)  # shape [batch_size]
        # 计算损失
        loss = loss_fn(outputs, targets)

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss
    return total_loss / len(dataloader)


def train():
    # 损失函数，优化器，更新参数

    # 确定 device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    dataloader = get_dataLoader(train=True)

    # 分词器
    tokenizer = JiebaTokenizer.from_vocab(config.MODEL_DIR / "vocab.txt")

    # 模型
    model = MYLSTMModel(
        vocab_size=tokenizer.vocab_size, papadding_idx=tokenizer.pad_token_index
    ).to(device)

    # 损失函数
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # 优化器
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
