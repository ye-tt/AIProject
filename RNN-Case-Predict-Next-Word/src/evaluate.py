import torch
import config
from model import MYRNNModel
from dataset import get_dataLoader
from predit import predit_batch
from tokenizer import JiebaTokenizer


def evaluate(model, dataloader, device):
    top1_acc_account = 0
    top5_acc_account = 0
    total_count = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)  # shape(batch_size, seq_len)
        targets = targets.tolist()  # shape(batch_size) e.g. [1,3,5]

        # 前向传播
        # shape(batch_size,vocab_size)
        outputs = model(inputs)
        # shape (batch_size,5) e.g.[[1,3,5,7,9],[3,4,5,7,2],[2,6,4,3,5]]
        top5_indexes_list = predit_batch(model, inputs)

        # 用top5_indexes_list 和target evaluate
        for target, top5_indexes in zip(targets, top5_indexes_list):
            total_count += 1
            if target == top5_indexes[0]:
                top1_acc_account += 1
            if target in top5_indexes:
                top5_acc_account += 1
    return top1_acc_account / total_count, top5_acc_account / total_count


def run_evaluate():
    # 准备资源
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载词表
    tokenizer = JiebaTokenizer.from_vocab(config.MODEL_DIR / "vocab.txt")

    # 模型
    model = MYRNNModel(vocab_size=tokenizer.vocab_size).to(device)
    # 用torch.load 加载 torch.save 的内容
    model.load_state_dict(torch.load(config.MODEL_DIR / "best.pth"))

    # 数据集
    dataloader = get_dataLoader(train=False)

    # 评估逻辑
    top1_acc, top5_acc = evaluate(model, dataloader, device)
    print("评估结果")
    print(f"top1_acc:{top1_acc}")
    print(f"top5_acc:{top5_acc}")


if __name__ == "__main__":
    run_evaluate()
