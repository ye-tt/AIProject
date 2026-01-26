import config
from model import CorrectionModel
from torch import optim
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

def read_vocab():
    with open(config.VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab_list = [line.strip() for line in f.readlines()]
    return vocab_list

def collate_fn(batch):
    # batch 是一个二元组列表，包含多个样本，每个样本是一个 (input_tensor, target_tensor) 元组 [(input_tensor1, target_tensor1),(input_tensor2, target_tensor2)....]
    # input_tensor= batch[0][0], target_tensor=batch[0][1]
    # input_tensor= [item[0] for item in batch], target_tensor=[item[1] for item in batch]
    input_tensors, target_tensors = zip(*batch)

    # 使用 pad_sequence 对输入和目标进行填充
    # input_tensors_padded.shape: [batch_size, max_seq_len]
    input_tensors_padded = pad_sequence(
        input_tensors, batch_first=True, padding_value=0
    )
    target_tensors_padded = pad_sequence(
        target_tensors, batch_first=True, padding_value=0
    )

    return input_tensors_padded, target_tensors_padded

def get_dataLoader(train=True):
    path = config.PROCESSED_DATA_DIR / ("train.jsonl" if train else "test.jsonl")
    data = pd.read_json(path, orient="records", lines=True).dropna().to_dict(
            orient="records"
        )
    train_x = [torch.tensor(item["source"]) for item in data]
    train_y = [torch.tensor(item["target"]) for item in data]
    data_set = list(zip(train_x, train_y))    
    # collate_fn: 是一个用于自定义如何将多个样本（samples）合并成一个 batch 的函数
    # 默认情况下，DataLoader 使用 torch.utils.data.default_collate 来自动把一批样本“堆叠”（stack）成张量,但前提是：所有样本的形状必须一致
    return DataLoader(
        data_set, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

def train_one_epoch(model, data_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for inputs, targets in tqdm(data_loader,desc="训练"):
        print(f"inputs shape: {inputs.shape}, targets shape: {targets.shape}")
        # 编码阶段
        # enconder_input shape (batch_size, src_seq_len)
        enconder_input = inputs.to(config.device)
        # targets shape (batch_size, trg_seq_len)
        targets = targets.to(config.device)   
        # context_vector shape [batch_size, hidden_size]
        context_vector = model.encoder(enconder_input) 

        # 解码阶段
        batch_size = context_vector.shape[0]
        decoder_inputs = targets[:, :-1]  # 去掉最后一个 <eos> 作为解码器输入
        decoder_targets = targets[:, 1:] # 去掉第一个 token 作为目标输出
        decoder_outputs = []
        # decoder_hidden_0 shape [1, batch_size, hidden_size]
        # LSTM 需要 (hidden, cell)
        decoder_hidden = (context_vector.unsqueeze(0), torch.zeros_like(context_vector.unsqueeze(0)))
        
        # 并行解码
        decoder_embed = model.decoder.embedding(decoder_inputs)  # [batch_size, seq_len, embed_dim]
        decoder_output, _ = model.decoder.lstm(decoder_embed, decoder_hidden)  # [batch_size, seq_len, hidden_size]
        decoder_outputs = model.decoder.liner(decoder_output)  # [batch_size, seq_len, vocab_size]
         # 重塑张量 [batch_size * seq_len, vocab_size]
        decoder_outputs = decoder_outputs.reshape(-1, decoder_outputs.shape[-1])
        # decoder_targets shape [batch_size * seq_len]
        decoder_targets = decoder_targets.reshape(-1)
        
        #CrossEntropyLoss  - Input: Shape :math:`(C)`, :math:`(N, C)`，
        #Target: If containing class indices, shape :math:`()`, :math:`(N)` 
        loss = loss_fn(decoder_outputs, decoder_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(data_loader)

    

def train():
    print("使用设备：",config.device)
    vocab_list = read_vocab()
    dataloader =get_dataLoader(train=True)
    model = CorrectionModel(vocab_size=len(vocab_list), padding_idx=config.PAD_IDEX).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARN_RATE)
    loss_fn = nn.CrossEntropyLoss(ignore_index=config.PAD_IDEX)
    best_loss = float("inf")
    for epoch in range(config.EPOCHS):
        print(f"Epoch {epoch+1}/{config.EPOCHS} - Training...")  
        loss = train_one_epoch(model,dataloader,optimizer,loss_fn)
        print(f"loss:{loss}")
        print(f"Epoch {epoch+1} completed.")
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), config.MODEL_DIR / "best.pth")  # .pt pytorch
            print("模型保存成功")
if __name__ == "__main__":
    train()    