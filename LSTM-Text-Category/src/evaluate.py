import torch
import config
from model import MYGRUModel
from dataset import get_dataLoader
from predict import predit_batch
from tokenizer import JiebaTokenizer


def evaluate(model, dataloader, device):
    for inputs, targets in dataloader:
        inputs = inputs.to(device)  # shape(batch_size, seq_len)
        targets = targets.tolist()  # shape(batch_size) e.g. [1,0,1，1]

        # 前向传播
        # shape(batch_size,1)
        outputs = model(inputs)
        # shape (batch_size) [0.5,0.3,0.8,0.1]
        batch_result = predit_batch(model, inputs)

        total_acc_count = 0
        acc_account = 0
        # 用top5_indexes_list 和target evaluate
        for target, pre_result in zip(targets, outputs):
            total_acc_count += 1
            result = 1 if pre_result > 0.5 else 0
            if target == result:
                acc_account += 1
    return acc_account / total_acc_count


def run_evaluate():
    # 准备资源
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载词表
    tokenizer = JiebaTokenizer.from_vocab(config.MODEL_DIR / "vocab.txt")

    # 模型
    model = MYGRUModel(
        vocab_size=tokenizer.vocab_size, papadding_idx=tokenizer.pad_token_index
    ).to(device)
    # 用torch.load 加载 torch.save 的内容
    model.load_state_dict(torch.load(config.MODEL_DIR / "best.pth"))

    # 数据集
    dataloader = get_dataLoader(train=False)

    # 评估逻辑
    accuracy = evaluate(model, dataloader, device)
    print("评估结果")
    print(f"acc:{accuracy}")


if __name__ == "__main__":
    run_evaluate()
