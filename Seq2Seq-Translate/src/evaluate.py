from nltk.translate.bleu_score import corpus_bleu
import torch
from tokenizer import ChineseTokenizer, EnglishTokenizer
import config
from model import TranslationModel
from dataset import get_dataLoader
from predict import predit_batch


def evaluate(model, test_dataloader, en_tokenizer, device):
    predictions = []  # 预测值
    references = []  # 参考答案
    for inputs, targets in test_dataloader:
        inputs = inputs.to(device)  # shape(batch_size, seq_len)
        # shape(batch_size) e.g. [[],[],[]...] 列表里每个元素长度相等，因为做了pad，还有sos,eos
        targets = targets.tolist()

        # shape (batch_size) [[],[],[]...] 这里每个元素的长度不同，因为已经根据<eos> 阶段其后的部分
        batch_result = predit_batch(model, inputs, en_tokenizer)

        # 准备计算bleu 的数据
        predictions.extend(batch_result)
        references.extend(
            [[tgt[1 : tgt.index(en_tokenizer.eos_token_index)]] for tgt in targets]
        )
        print(f"Sample prediction: {batch_result}")
        print(f"Sample reference: {targets}")
        print(f"references: {references}")
        print(f"predictions: {predictions}")
        print(corpus_bleu(references, predictions))
        break
    return corpus_bleu(references, predictions)


def run_evaluate():
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载词表
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODEL_DIR / "zh_vocab.txt")
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODEL_DIR / "en_vocab.txt")
    print("词表加载成功")
    # 模型
    model = TranslationModel(
        zh_tokenizer.vocab_size,
        en_tokenizer.vocab_size,
        zh_padding_idx=zh_tokenizer.pad_token_index,
        en_padding_idx=en_tokenizer.pad_token_index,
    ).to(device)

    model.load_state_dict(torch.load(config.MODEL_DIR / "best.pth"))
    print("模型加载成功")
    # 测试数据集
    test_dataloader = get_dataLoader(train=False)
    # 评估逻辑
    bleu_score = evaluate(model, test_dataloader, en_tokenizer, device)
    print("评估结果")
    print(f"BLEU Score: {bleu_score * 100:.2f}")


if __name__ == "__main__":
    run_evaluate()
