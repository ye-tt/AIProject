import config
import torch
from model import CorrectionModel
from train import read_vocab,get_dataLoader


def predit_batch(model, inputs,sos_token_index,eos_token_index):
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
        context_vector = model.encoder(inputs)
        batch_size = inputs.shape[0]
        

        # 解码
        # decoder_hidden shape [1, batch_size, hidden_size]， RNN  h_0 的形状是 (num_layers * num_directions, batch, hidden_size)
        decoder_hidden = (
            context_vector.unsqueeze(0),
            torch.zeros_like(context_vector.unsqueeze(0)).to(config.device)
        )

        # <sos>, x, xx, xxxxx... <sos> 作为第一个时间步的输入
        decoder_input = torch.full(
            (batch_size, 1),
            fill_value=sos_token_index,
            device=config.device
        )

        # 预测结果缓存
        generated = []

        # 记录每个样本是否已经结束
        is_finished = torch.full((batch_size,), fill_value=False).to(config.device)

        # 自回归生成，直到生成<eos>或者达到最大长度
        for i in range(config.MAX_SEQ_LEN):
            decoder_output, decoder_hidden = model.decoder(
                decoder_input, decoder_hidden
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
                == eos_token_index
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
            if eos_token_index in sentence:
                # 取出第一个eos 的位置
                eos_index = sentence.index(eos_token_index)
                # 截断序列，保留到eos 位置
                generated_list[index] = sentence[:eos_index]
        return generated_list


def predict(text, model, word2index,unk_token_index,sos_token_index,eos_token_index ):
    # 字符串分词，转成id, 封装成tensor
    tokens = list(text)   
    indexs = [word2index.get(token, unk_token_index) for token in tokens]
    print(f"原始文本: {text}")
    print(f"分词结果: {tokens}")
    print(f"索引结果: {indexs}")
    # indexs(seq_len)
    # input_tensor.shape[batch_size, seq_len]
    input_tensor = torch.tensor([indexs], dtype=torch.long).to(config.device)
    print("input_tensor.shape:", input_tensor.shape)

    # 预测逻辑 batch_result 是一个链表，shape (batch_size)，这里batch_size=1
    batch_result = predit_batch(model, input_tensor,sos_token_index,eos_token_index)
    print("batch_result", batch_result)
    # id 转 字符
    index2word = {index: word for word, index in word2index.items()}
    batch_result = [
        [index2word.get(index, config.NUK_TOKEN) for index in sentence]
        for sentence in batch_result
    ]
    print(batch_result)
    return batch_result[0]  # 返回第一个样本的预测结果


def run_predict():
    # 加载分词器
    vocab_list = read_vocab()
    word2index = {word: index for index, word in enumerate(vocab_list)}
    unk_token_index= word2index.get(config.NUK_TOKEN)
    sos_token_index = word2index.get(config.SOS_TOKEN)  
    eos_token_index = word2index.get(config.ESO_TOKEN)  
    print("词表加载成功")
    # 模型
    dataloader =get_dataLoader(train=True)
    model = CorrectionModel(vocab_size=len(vocab_list), padding_idx=config.PAD_IDEX).to(config.device)
    model.load_state_dict(torch.load(config.MODEL_DIR / "best.pth"))
    print("模型加载成功")
    # 测试数据集

    print("输入原始语句（输入q 或者quit推出）")
    while True:
        user_input = input("")
        if user_input in ["q", "quit"]:
            break
        if user_input.strip() == "":
            continue
        result = predict(user_input, model,word2index,unk_token_index,sos_token_index,eos_token_index)
        print(f"预测结果：{result}")


def evaluate_on_testset():
    vocab_list = read_vocab()
    word2index = {word: index for index, word in enumerate(vocab_list)}
    index2word = {index: word for word, index in word2index.items()}
    
    sos_token_index = word2index.get(config.SOS_TOKEN)  
    eos_token_index = word2index.get(config.ESO_TOKEN)
    
    print(f"SOS索引: {sos_token_index}, EOS索引: {eos_token_index}")
    print(f"PAD索引: {config.PAD_IDEX}")
    
    test_dataloader = get_dataLoader(train=False)
    model = CorrectionModel(vocab_size=len(vocab_list), padding_idx=config.PAD_IDEX).to(config.device)
    model.load_state_dict(torch.load(config.MODEL_DIR / "best.pth", map_location=config.device))
    model.eval()
    
    correct = 0
    total = 0
    exact_match = 0
    
    # 添加详细的调试信息
    for batch_idx, (inputs, targets) in enumerate(test_dataloader):
        inputs = inputs.to(config.device)
        batch_result = predit_batch(model, inputs, sos_token_index, eos_token_index)
        
        # 比较预测结果和真实标签
        for i, (pred, target) in enumerate(zip(batch_result, targets)):
            target_list = target.tolist()
            original_target = target_list.copy()
            
            # 去掉 padding
            if config.PAD_IDEX in target_list:
                target_list = [idx for idx in target_list if idx != config.PAD_IDEX]
            
            # 去掉 <sos> 和 <eos>
            if sos_token_index in target_list:
                target_list.remove(sos_token_index)
            if eos_token_index in target_list:
                eos_idx = target_list.index(eos_token_index)
                target_list = target_list[:eos_idx]
            
            # 调试信息 - 打印更多样本
            if batch_idx == 0 and i < 5:
                input_list = inputs[i].tolist()
                input_text = ''.join([index2word.get(idx, '?') for idx in input_list if idx != config.PAD_IDEX])
                
                print(f"\n{'='*50}")
                print(f"样本 {i}:")
                print(f"输入文本: {input_text}")
                print(f"输入索引: {[idx for idx in input_list if idx != config.PAD_IDEX]}")
                print(f"原始目标: {original_target}")
                print(f"处理后目标: {target_list}")
                print(f"预测索引: {pred}")
                print(f"预测长度: {len(pred)}, 目标长度: {len(target_list)}")
                print(f"预测文本: {''.join([index2word.get(idx, '?') for idx in pred])}")
                print(f"目标文本: {''.join([index2word.get(idx, '?') for idx in target_list])}")
                print(f"完全匹配: {pred == target_list}")
            
            # 完全匹配
            if pred == target_list:
                exact_match += 1
            
            # 字符级准确率
            min_len = min(len(pred), len(target_list))
            for j in range(min_len):
                if pred[j] == target_list[j]:
                    correct += 1
            total += len(target_list)
        
        if batch_idx >= 0:  # 只看第一个batch
            break
    
    print(f"\n{'='*50}")
    print(f"完全匹配准确率: {exact_match}/{len(test_dataloader.dataset)*config.BATCH_SIZE if batch_idx == 0 else (batch_idx+1)*config.BATCH_SIZE}")
    print(f"字符级准确率: {correct/total*100:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    # run_predict()
    evaluate_on_testset()
