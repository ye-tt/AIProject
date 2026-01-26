import jieba
import config
import torch
from model import MYRNNModel
from tokenizer import JiebaTokenizer


def predit_batch(model, inputs):
    model.eval()
    with torch.no_grad():
        output = model(inputs)
        # output.shape(batch_size,vocab_size)

    # 返回指定维度 前k 个index, value, 取vocab_size 维度里概率值最大的下标
    top5_indexes = torch.topk(output, k=5).indices
    # top5_indexes.shape(batch_size,5)

    # 将 id 转成词
    top5_indexes_list = top5_indexes.tolist()
    return top5_indexes_list


def predict(text, model, tokenizer, device):
    # 字符串分词，转成id, 封装成tensor

    # indexs(seq_len)
    indexs = tokenizer.encode(text)
    # input_tensor.shape[batch_size, seq_len]
    input_tensor = torch.tensor([indexs], dtype=torch.long).to(device)

    # 预测逻辑
    top5_indexes_list = predit_batch(model, input_tensor)
    top5_tokens = [tokenizer.index2word[index] for index in top5_indexes_list[0]]
    return top5_tokens


def run_predict():
    # 准备各种资源
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载词表
    tokenizer = JiebaTokenizer.from_vocab(config.MODEL_DIR / "vocab.txt")

    # 模型
    model = MYRNNModel(vocab_size=tokenizer.vocab_size).to(device)
    # 用torch.load 加载 torch.save 的内容
    model.load_state_dict(torch.load(config.MODEL_DIR / "best.pth"))
    print("欢迎使用输入法模型（输入q 或者quit推出）")
    input_history = ""
    while True:
        user_input = input("> ")
        if user_input in ["q", "quit"]:
            break
        if user_input.strip() == "":
            continue
        input_history += user_input
        print(f"输入历史：{input_history}")
        top5_tokens = predict(input_history, model, tokenizer, device)
        print(f"预测结果：{top5_tokens}")


if __name__ == "__main__":
    run_predict()
