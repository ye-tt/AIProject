from os import truncate
import config
import torch
from model import MYBertModel
from transformers import AutoTokenizer


def predit_batch(model, inputs):
    """
    对一个批次输入进行预测。
    :param input_tensor: 输入张量 (batch_size, seq_len)
    :param model: 模型
    :return: 概率列表
    """
    model.eval()
    with torch.no_grad():
        output = model(**inputs)
        # output.shape(batch_size)

        # 将 output 通过sigmod 转成概率
        batch_result = torch.sigmoid(output).tolist()
        # 确保返回的是列表（处理单个样本的情况）
        if not isinstance(batch_result, list):
            batch_result = [batch_result]
        return batch_result


def predict(text, model, tokenizer, device):
    #  shape[batch_size, seq_len]
    # (input_ids,token_type_ids,attention_mask,labels)
    inputs = tokenizer(
        text,
        padding="max_length",
        max_length=config.MAX_SEQ_LEN,
        truncation=True,
        return_tensors="pt",
    )
    ## to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # 预测逻辑 batch_result 是一个链表，shape (batch_size)
    batch_result = predit_batch(model, inputs)
    return batch_result[0]


def run_predict():
    # 准备各种资源
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载词表
    tokenizer = AutoTokenizer.from_pretrained(
        config.PRE_TRAINED_DIR / "bert-base-chinese"
    )

    # 模型
    model = MYBertModel().to(device)
    # 用torch.load 加载 torch.save 的内容
    model.load_state_dict(torch.load(config.MODEL_DIR / "best.pth"))
    print("欢迎使用情感分析模型（输入q 或者quit推出）")
    while True:
        user_input = input("> ")
        if user_input in ["q", "quit"]:
            break
        if user_input.strip() == "":
            continue
        result = predict(user_input, model, tokenizer, device)
        if result > 0.5:
            print(f"预测结果：正向, result={result}")
        else:
            print(f"预测结果：负向")


if __name__ == "__main__":
    run_predict()
