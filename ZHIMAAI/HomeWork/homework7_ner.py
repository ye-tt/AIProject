import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datasets import load_dataset

class NERDataset(Dataset):
    """NER数据集"""
    def __init__(self, sentences, ner_labels, vocab, label_vocab, window_size=3):
        self.sentences = sentences
        self.ner_labels = ner_labels
        self.vocab = vocab
        self.label_vocab = label_vocab
        self.window_size = window_size
        self.features, self.targets = self._prepare_data()
    
    def _create_context_features(self, sentence):
        """创建上下文特征"""
        window_size = self.window_size
        padded = ['<PAD>'] * window_size + list(sentence) + ['<PAD>'] * window_size
        features = []
        
        for i in range(window_size, len(padded) - window_size):
            context_indices = []
            for j in range(i - window_size, i + window_size + 1):
                char = padded[j]
                idx = self.vocab.char2idx.get(char, self.vocab.char2idx['<UNK>'])
                one_hot = np.zeros(self.vocab.vocab_size)
                one_hot[idx] = 1
                context_indices.extend(one_hot)
            features.append(context_indices)
        
        return features
    
    def _prepare_data(self):
        """准备数据"""
        features = []
        targets = []
        
        for sentence, labels in zip(self.sentences, self.ner_labels):
            sentence_features = self._create_context_features(sentence)
            sentence_labels = [self.label_vocab.label2idx[label] for label in labels]
            
            features.extend(sentence_features)
            targets.extend(sentence_labels)
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(targets, dtype=torch.long)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class NERVocabulary:
    """NER词汇表"""
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.label2idx = {}
        self.idx2label = {}
        
    def build_char_vocab(self, sentences, min_freq=1):
        """构建字符词汇表"""
        char_counts = Counter()
        for sentence in sentences:
            char_counts.update(sentence)
        
        special_tokens = ['<PAD>', '<UNK>']
        common_chars = [char for char, count in char_counts.items() if count >= min_freq]
        
        all_chars = special_tokens + common_chars
        self.char2idx = {char: idx for idx, char in enumerate(all_chars)}
        self.idx2char = {idx: char for idx, char in enumerate(all_chars)}
        self.vocab_size = len(all_chars)
        
        print(f"字符词汇表大小: {self.vocab_size}")
    
    def build_label_vocab(self, all_labels):
        """构建标签词汇表"""
        unique_labels = sorted(set(all_labels))
        self.label2idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx2label = {idx: label for idx, label in enumerate(unique_labels)}
        self.num_labels = len(unique_labels)
        
        print(f"NER标签数量: {self.num_labels}")
        print(f"标签列表: {list(self.label2idx.keys())}")

class MLPNERModel(nn.Module):
    """基于MLP的NER模型"""
    
    def __init__(self, input_size, hidden_sizes=[256, 128], num_classes=10, dropout=0.3):
        super(MLPNERModel, self).__init__()
        
        # 构建MLP层
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.mlp = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.mlp(x)

class NERSystem:
    """NER系统主类"""
    
    def __init__(self, window_size=3, hidden_sizes=[256, 128], dropout=0.3, lr=0.001):
        self.window_size = window_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.lr = lr
        
        self.vocab = NERVocabulary()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def load_data_set(self):
        try:
            dataset = load_dataset("Aunderline/CMeEE-V2")
            print("数据集加载成功！")
            train_data = dataset["train"]
            test_data= dataset["test"]
            self.train_data = train_data
            self.test_data = test_data
        except Exception as e:
            print(f"加载失败: {e}")        

    def prepare_training_data(self,dataSet):
        """准备NER训练数据"""
       
        
        # print(f"X_train: {train_data}")
        sentences = [example['text'] for example in dataSet]
        ner_labels = []
        # print(train_data[0])
        for data_example in dataSet:
            data_text = data_example.get('text')
            entities = data_example.get('entities', [])
            # print(f"text: {data_text}")
            # print(f"entities: {entities}")
            # 创建BIO标签序列
            bio_tags = ['O'] * len(data_text)
            for entity in entities:
                start = entity['start_idx']
                end = entity['end_idx']
                ent_type = entity['type']
                if start < len(bio_tags):
                    bio_tags[start] = f'B-{ent_type}'
                    for i in range(start + 1, min(end, len(bio_tags))):
                        bio_tags[i] = f'I-{ent_type}'           
            ner_labels.append(bio_tags)

            # print(f"ner_labels: {ner_labels}")
            # print(f"sentences: {sentences}")
            # sentences_list = [list(text) for text in sentences]
            return sentences, ner_labels
    
    def train(self, epochs=100, batch_size=16, save_path='ner_model.pth'):
        """训练NER模型"""
        # 准备数据
        self.load_data_set()
        sentences, ner_labels = self.prepare_training_data(self.train_data)
        print('ner_labels......',ner_labels)
        # 构建词汇表
        self.vocab.build_char_vocab(sentences)
        
        # 收集所有标签
        all_labels = []
        for labels in ner_labels:
            all_labels.extend(labels)
        self.vocab.build_label_vocab(all_labels)
        
        # 创建数据集
        dataset = NERDataset(sentences, ner_labels, self.vocab, self.vocab, self.window_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 计算输入维度
        input_size = (2 * self.window_size + 1) * self.vocab.vocab_size
        
        # 初始化模型
        self.model = MLPNERModel(
            input_size=input_size,
            hidden_sizes=self.hidden_sizes,
            num_classes=self.vocab.num_labels,
            dropout=self.dropout
        ).to(self.device)
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        # 训练
        print("开始训练NER模型...")
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for features, labels in dataloader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 统计
                total_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # 计算指标
            avg_loss = total_loss / total
            accuracy = correct / total
            
            losses.append(avg_loss)
            accuracies.append(accuracy)
            
            scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | "
                      f"Accuracy: {accuracy:.2%} | LR: {current_lr:.6f}")
        
        # 保存模型
        self.save_model(save_path)
        
        # 绘制曲线
        self.plot_training_curve(losses, accuracies)
        
        return losses, accuracies


    ## predict的时候不需要反向传播，因此不需要loss， no_grad
    def predict(self, sentence):
        """预测NER标签"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        self.model.eval()
        
        # 创建特征
        window_size = self.window_size
        padded = ['<PAD>'] * window_size + list(sentence) + ['<PAD>'] * window_size
        features = []
        
        for i in range(window_size, len(padded) - window_size):
            context_indices = []
            for j in range(i - window_size, i + window_size + 1):
                char = padded[j]
                idx = self.vocab.char2idx.get(char, self.vocab.char2idx['<UNK>'])
                one_hot = np.zeros(self.vocab.vocab_size)
                one_hot[idx] = 1
                context_indices.extend(one_hot)
            features.append(context_indices)
        
        # 预测
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(features_tensor)
            _, predictions = torch.max(outputs, 1)
            predictions = predictions.cpu().numpy()
        
        # 转换为标签
        labels = [self.vocab.idx2label[p] for p in predictions]
        
        return labels
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab': self.vocab,
            'window_size': self.window_size,
            'hidden_sizes': self.hidden_sizes,
            'dropout': self.dropout
        }, path)
        print(f"NER模型已保存到: {path}")
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.vocab = checkpoint['vocab']
        self.window_size = checkpoint['window_size']
        self.hidden_sizes = checkpoint['hidden_sizes']
        self.dropout = checkpoint['dropout']
        
        # 计算输入维度
        input_size = (2 * self.window_size + 1) * self.vocab.vocab_size
        
        # 初始化模型
        self.model = MLPNERModel(
            input_size=input_size,
            hidden_sizes=self.hidden_sizes,
            num_classes=self.vocab.num_labels,
            dropout=self.dropout
        ).to(self.device)
        
        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"NER模型已从 {path} 加载")
    

    def evaluate(self, sentences, ner_labels, max_examples=10):
        if self.model is None:
            raise ValueError("模型未初始化或未加载，请先load_model()")
        self.model.eval()
        total = 0
        correct = 0

        for i, (sent_chars, true_tags) in enumerate(zip(sentences, ner_labels)):
            if i >= max_examples and max_examples > 0:
                break
            pred_tags = self.predict("".join(sent_chars))
            # 保证长度一致
            L = min(len(pred_tags), len(true_tags))
            for p, t in zip(pred_tags[:L], true_tags[:L]):
                if p == t:
                    correct += 1
                total += 1

            # 打印若干示例用于人工检查
            if i < 5:
                print("句子:", "".join(sent_chars))
                print("真实:", true_tags[:L])
                print("预测:", pred_tags[:L])
                print("-" * 40)

        accuracy = correct / total if total > 0 else 0.0
        print(f"Token-level Accuracy (on {min(len(sentences), max_examples) if max_examples>0 else len(sentences)} samples): {accuracy:.4f}")
        return accuracy
    def plot_training_curve(self, losses, accuracies):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(losses, 'b-', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('NER Training Loss')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(accuracies, 'r-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('NER Training Accuracy')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def test_ner_system():
    """测试NER系统"""
    print("=" * 60)
    print("命名实体识别(NER)系统测试")
    print("=" * 60)
    
    # 创建NER系统
    ner_system = NERSystem(
        window_size=2,
        hidden_sizes=[128, 64],
        dropout=0.2,
        lr=0.001
    )
    
    # 训练模型
    ner_system.train(epochs=50, batch_size=8, save_path='ner_model.pth')
    
    # 测试

    test_sentences,test_labels = ner_system.prepare_training_data(ner_system.test_data)
    print("\n" + "=" * 60)
    print("NER识别结果:")
    print("=" * 60)
    

    ner_system.evaluate( test_sentences, test_labels, max_examples=10)



# ner = NERSystem()
# ner.train()

test_ner_system()

