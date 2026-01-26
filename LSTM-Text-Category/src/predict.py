import jieba
import config
import torch
from model import MYGRUModel
from tokenizer import JiebaTokenizer


def predit_batch(model, inputs):
    """
    对一个批次输入进行预测。
    :param input_tensor: 输入张量 (batch_size, seq_len)
    :param model: 模型
    :return: 概率列表
    """
    model.eval()
    with torch.no_grad():
        output = model(inputs)
        # output.shape(batch_size)

        # 将 output 通过sigmod 转成概率
        batch_result = torch.sigmoid(output).tolist()
        return batch_result


def predict(text, model, tokenizer, device):
    # 字符串分词，转成id, 封装成tensor

    # indexs(seq_len)
    indexs = tokenizer.encode(text, config.SEQ_LEN)
    # input_tensor.shape[batch_size, seq_len]
    input_tensor = torch.tensor([indexs], dtype=torch.long).to(device)
    print("input_tensor.shape:", input_tensor.shape)
    print("input_tensor", input_tensor)

    # 预测逻辑 batch_result 是一个链表，shape (batch_size)
    batch_result = predit_batch(model, input_tensor)
    print("batch_result", batch_result)
    return batch_result[0]


def run_predict():
    # 准备各种资源
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载词表
    tokenizer = JiebaTokenizer.from_vocab(config.MODEL_DIR / "vocab.txt")

    # 模型
    model = MYGRUModel(
        vocab_size=tokenizer.vocab_size, papadding_idx=tokenizer.pad_token_index
    ).to(device)
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
