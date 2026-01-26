from os import truncate
import config
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


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
        logits = output.logits   #[batch_size, 2]
        # 将 output 通过sigmod 转成概率
        batch_result = torch.argmax( logits, dim=-1)
        return batch_result.tolist()


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
    if isinstance(batch_result, list):
        return batch_result[0]
    return batch_result


def run_predict():
    # 准备各种资源
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载词表
    tokenizer = AutoTokenizer.from_pretrained(
        config.PRE_TRAINED_DIR / "bert-base-chinese"
    )

    # 模型
    model = AutoModelForSequenceClassification.from_pretrained(config.MODEL_DIR).to(
        device
    )
    print("欢迎使用情感分析模型（输入q 或者quit推出）")
    while True:
        user_input = input("> ")
        if user_input in ["q", "quit"]:
            break
        if user_input.strip() == "":
            continue
        result = predict(user_input, model, tokenizer, device)
        if result == 1:
            print(f"预测结果：正向, result={result}")
        else:
            print(f"预测结果：负向")


if __name__ == "__main__":
    run_predict()
