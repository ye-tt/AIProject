import torch
import config
from dataset import get_dataLoader
from predict import predit_batch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from tqdm import tqdm


def evaluate(model, test_dataloader, device):
    for batch in tqdm(test_dataloader, desc="Testing"):
        # batch(input_ids,token_type_ids,attention_mask,labels)
        labels = batch.pop("labels")
        inputs = {k:v.to(device) for k, v in batch.items()}
        
        # 前向传播
        batch_result = predit_batch(model, inputs)

        total_acc_count = 0
        acc_account = 0
        # 用top5_indexes_list 和target evaluate
        for target, pre_result in zip(labels, batch_result):
            total_acc_count += 1
            if target == pre_result:
                acc_account += 1
    return acc_account / total_acc_count


def run_evaluate():
    # 准备资源
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载词表
    tokenizer = AutoTokenizer.from_pretrained(
        config.PRE_TRAINED_DIR / "bert-base-chinese"
    )

    # 模型
    model =AutoModelForSequenceClassification.from_pretrained(config.MODEL_DIR).to(device)

    # 数据集
    test_dataloader = get_dataLoader(train=False)

    # 评估逻辑
    accuracy = evaluate(model, test_dataloader, device)
    print("评估结果")
    print(f"acc:{accuracy}")


if __name__ == "__main__":
    run_evaluate()
