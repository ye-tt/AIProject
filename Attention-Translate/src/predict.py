import config
import torch
from model import TranslationModel
from tokenizer import ChineseTokenizer, EnglishTokenizer


def predit_batch(model, inputs, en_tokenizer):
    """
    对一个批次输入进行预测。
    :param input_tensor: 输入张量 (batch_size, seq_len)
    :param model: 模型
    :return: 预测结果[[*,*,*],[*,*,*,*],[*,*]...]
    """
    model.eval()
    with torch.no_grad():
        # 编码
        # context_vector.shape [batch_size, hidden_size]
        encoder_outputs, context_vector = model.encoder(inputs)
        batch_size = inputs.shape[0]
        device = inputs.device

        # 解码
        # decoder_hidden shape [1, batch_size, hidden_size]， RNN  h_0 的形状是 (num_layers * num_directions, batch, hidden_size)
        decoder_hidden = context_vector.unsqueeze(0)

        # 英文<sos>, x, xx, xxxxx... <sos> 作为第一个时间步的输入
        decoder_input = torch.full(
            (batch_size, 1),
            fill_value=en_tokenizer.sos_token_index,
        )

        # 预测结果缓存
        generated = []

        # 记录每个样本是否已经结束
        is_finished = torch.full((batch_size,), fill_value=False)

        # 自回归生成，直到生成<eos>或者达到最大长度
        for i in range(config.MAX_SEQ_LENGTH):
            decoder_output, decoder_hidden = model.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # decoder_output shape : [batch_size, 1, vocab_size]

            # 选择概率最大的词作为下一个输入
            next_token_idexes = decoder_output.argmax(dim=-1)  # shape: [batch_size, 1]
            # 保存预测结果
            generated.append(next_token_idexes)
            # 更新输入 （decoder_input）
            decoder_input = next_token_idexes

            # 判断是否全部生成了<eos>
            # 判断是否结束
            # is_finished | = next_token_idexes.squeeze(1) == en_tokenizer.eos_token_index
            is_finished = (
                is_finished | next_token_idexes.squeeze(1)
                == en_tokenizer.eos_token_index
            )
            if is_finished.all():
                break
        # 处理预测结果
        # generated: [tensor[batch_size, 1],tensor[batch_size, 1...], seq_len 个[batch_size, 1] -> [batch_size, seq_len]
        # torch.cat 在dim=1 维度上拼接 ->[batch_size, seq_len]
        generated_tensor = torch.cat(generated, dim=1)
        generated_list = generated_tensor.tolist()  # [[*,*,**],[*,*,*]...]

        # 去掉每个序列中的<eos>及其后的部分
        for index, sentence in enumerate(generated_list):
            # <eos> 是否在句子中
            if en_tokenizer.eos_token_index in sentence:
                # 取出第一个eos 的位置
                eos_index = sentence.index(en_tokenizer.eos_token_index)
                # 截断序列，保留到eos 位置
                generated_list[index] = sentence[:eos_index]
        return generated_list


def predict(text, model, zh_tokenizer, en_tokenizer, device):
    # 字符串分词，转成id, 封装成tensor

    # indexs(seq_len)
    indexs = zh_tokenizer.encode(text, add_sos_eos=False)
    print("indexs:", indexs)
    # input_tensor.shape[batch_size, seq_len]
    input_tensor = torch.tensor([indexs], dtype=torch.long).to(device)
    print("input_tensor.shape:", input_tensor.shape)

    # 预测逻辑 batch_result 是一个链表，shape (batch_size)，这里batch_size=1
    batch_result = predit_batch(model, input_tensor, en_tokenizer)
    print("batch_result", batch_result)
    return en_tokenizer.decode(batch_result[0])  # 返回第一个样本的预测结果


def run_predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODEL_DIR / "zh_vocab.txt")
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODEL_DIR / "en_vocab.txt")
    # 加载模型
    model = TranslationModel(
        zh_tokenizer.vocab_size,
        en_tokenizer.vocab_size,
        zh_tokenizer.pad_token_index,
        en_tokenizer.pad_token_index,
    ).to(device)
    model.load_state_dict(torch.load(config.MODEL_DIR / "best.pth"))

    print("欢迎使用中英文翻译模型（输入q 或者quit推出）")
    while True:
        user_input = input("zhong文：")
        if user_input in ["q", "quit"]:
            break
        if user_input.strip() == "":
            continue
        result = predict(user_input, model, zh_tokenizer, en_tokenizer, device)
        print(f"预测结果：{result}")


if __name__ == "__main__":
    run_predict()
