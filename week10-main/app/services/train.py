import torch
from app.core.config import settings
from bert_triage_model import BertTriageModel
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader




def load_triage_dataloader():
        dataset_dict = load_dataset("csv", data_files=str(settings.me_data_dir / "儿科5-14000.csv"), encoding='gb18030')
        train_dataset = dataset_dict['train']
        sample_ratio = 0.1  # 采样10%的数据
        sample_size = int(len(train_dataset) * sample_ratio)
        train_dataset = train_dataset.shuffle(seed=42).select(range(sample_size))

        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")
        train_dataset = train_dataset.remove_columns(['title', 'answer'])
        unique_departments = train_dataset.unique('department')
        label_to_id = {dept: idx for idx, dept in enumerate(unique_departments)}
        id_to_label = {idx: dept for dept, idx in label_to_id.items()}

        def add_labels(example):
            example['label'] = label_to_id[example['department']]
            return example
        train_dataset = train_dataset.map(add_labels)

        def batch_encode(batch):
            # example:{label:[1,0,1,0],review:['******','******','******','******']}
            inputs = tokenizer(batch['ask'], padding='max_length', max_length=128, truncation=True)
            return inputs

        # departments=  test_dataset.unique("department")
        train_dataset = train_dataset.map(batch_encode, batched=True,
                                          remove_columns=['ask', 'department'])
        train_dataset.save_to_disk('data/processed')
        train_dataset.set_format(type='torch')
        dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        # print(f"Dataset size: {len(train_dataset)}")
        # print(f"Dataset columns: {train_dataset.column_names}")
        # print(f"First batch shape:")
        # first_batch = next(iter(dataloader))
        # for key, value in first_batch.items():
        #     print(f"  {key}: {value.shape}")
        return dataloader, len(unique_departments), label_to_id, id_to_label




def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    total_loss = 0
    model.train()
    for batch in tqdm(dataloader, desc='训练'):
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
        # print("inputs:",inputs)
        labels = batch['label'].to(device, dtype=torch.long)
        # print("labels:", labels)
        # print("labels: shape", labels.shape)
        outputs = model(input_ids =inputs['input_ids'], attention_mask=inputs['attention_mask'], token_type_ids=inputs['token_type_ids'])

        # print("outputs:",outputs)
        # print("outputs: shape", labels.shape)

        # outputs.shape: [batch_size]
        loss = loss_fn(outputs, labels)
        # break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_val)

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def train():
    # 1. 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 2. 数据
    dataloader, num_classes, label_to_id, id_to_label = load_triage_dataloader()
    print(f"Number of classes: {num_classes}")
    print(f"Label mapping: {label_to_id}")
    # 3. 分词器
    tokenizer = AutoTokenizer.from_pretrained(settings.bert_triage_model)
    # 4. 模型
    model = BertTriageModel(num_classes).to(device)
    # 5. 损失函数
    loss_fn = torch.nn.CrossEntropyLoss()
    # 6. 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learn_rate,weight_decay=settings.weight_decay)
    # 7. TensorBoard Writer

    best_loss = float('inf')
    for epoch in range(1, settings.epochs+ 1):
        print(f'========== Epoch {epoch} ==========')
        loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        print(f'Loss: {loss:.4f}')

        # 保存模型
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), settings.storage_dir / 'best.pt')
            print('保存模型')
if __name__ == "__main__":
    train()