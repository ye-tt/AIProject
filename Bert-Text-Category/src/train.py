import torch
from dataset import get_dataLoader
from model import MYBertModel
import config
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
from transformers import AutoTokenizer


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    # batch(input_ids,token_type_ids,attention_mask,labels)
    for batch in tqdm(dataloader, desc="Training"):
        inputs = {k: v.to(device) for k, v in batch.items()}
        # inputs shape (batch_size, seq_len)
        lables = inputs.pop("labels").to(dtype=torch.float32)

        # 前向传播
        outputs = model(**inputs)
        # print(outputs.shape, lables.shape)
        # 计算损失
        loss = loss_fn(outputs, lables)

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

    # 模型
    model = MYBertModel().to(device)

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
