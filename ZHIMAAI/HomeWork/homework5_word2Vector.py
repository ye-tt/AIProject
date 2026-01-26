import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier

class Word2VectorModel:
    def __init__(self, corpus, window_size,embedding_dim=10):
        self.corpus = corpus
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.build_vocab()

    def build_vocab(self):
        words=[]
        for sentence in self.corpus:
            sentence = sentence.lower().replace('.', '')
            words.extend(sentence.split())
        self.vocab = sorted(set(words))
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.vocab)
        print(f"词汇表大小: {self.vocab_size}, 词汇表: {self.vocab}")
        
    def build_Xtrain_Vector(self, X_context):
        # 使用CountVectorizer将文本转换为特征向量
        context_vectorizer = CountVectorizer(vocabulary=self.vocab, token_pattern=r'(?u)\b\w+\b')
        X_encoded = context_vectorizer.fit_transform(X_context) 
        self.context_vectorizer = context_vectorizer     
        return X_encoded

    # 创建X 和y
    def create_corpus(self):
        pass

    def train_and_fit(self, X_context, y_center): 
        # 构建训练特征向量       
        X_encoded= self.build_Xtrain_Vector(X_context)
        
        # 使用逻辑回归训练模型
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_encoded, y_center)
        self.model=model        

    def prediction(self, X_Test_context):
        predicted_idx_list=[]
        for context in X_Test_context:
            # 将测试上下文转换为特征向量
            X_test = self.context_vectorizer.transform([context])
            # print("predictionXtest: ",X_test)  
            predicted_idx = self.model.predict(X_test) [0]
            predicted_idx_list.append(predicted_idx)
            # predicted_word = self.idx_to_word[predicted_idx]
            # predicted_words.append(predicted_word)
        return predicted_idx_list
                    
    

class CBOWModel(Word2VectorModel):   
    def create_corpus(self):
        X_context=[]
        y_center=[]
        for doc in self.corpus:
            words = doc.lower().replace('.', '').split()
            for i in range(len(words)):
                y_center.append(self.word_to_idx[words[i]])
                ctx = [words[j] for j in range(i-self.window_size, i+self.window_size+1) 
                        if j != i and 0 <= j < len(words)]
                if ctx:
                    X_context.append(" ".join(ctx))                 
        # print("X_context: ",X_context) 
        # print("y_center： ",y_center) 
        return X_context, y_center
    def prediction(self, X_Test_context):
        predicted_idx_list = super().prediction(X_Test_context)
        predicted_words = [self.idx_to_word[idx] for idx in predicted_idx_list]
        for context, predicted_word in zip(X_Test_context, predicted_words):
            print(f"上下文: '{context}' -> 预测中心词: '{predicted_word}'")

class SkipGramModel(Word2VectorModel):  
    def create_corpus(self):
        X_center=[]
        y_context=[]       
        for doc in self.corpus:
            words = doc.lower().replace('.', '').split()
            for i in range(len(words)):
                context_labels = np.zeros(self.vocab_size)
                X_center.append(words[i])
                for j in range(i-self.window_size, i+self.window_size+1):
                    if j != i and 0 <= j < len(words):
                        context_idx = self.word_to_idx[words[j]]
                        context_labels[context_idx] = 1 
                y_context.append(context_labels)        
        # print("X_center: ",X_center) 
        # print("y_context ",y_context)        
        return X_center, np.array(y_context)
    
    def train_and_fit(self, X_center, y_context):       
        # 使用CountVectorizer将中心词转换为特征向量
        X_encoded= super().build_Xtrain_Vector(X_center)
        print(f"训练数据形状: X={X_encoded.shape}, y={y_context.shape}")
        
        # 使用MLP神经网络训练多输出模型
        model = MLPClassifier(
            hidden_layer_sizes=(self.embedding_dim,),
            max_iter=1000,
            random_state=42,
            learning_rate_init=0.01
        )
        
        # 由于是多标签问题，我们需要使用特殊处理
        # 这里我们训练多个二分类器
        multi_model = MultiOutputClassifier(model, n_jobs=-1)
        multi_model.fit(X_encoded, y_context)        
        self.model=multi_model

    def prediction(self, X_Test_context):
        predicted_idx_list = super().prediction(X_Test_context)
        context_idxs = [[j for j, val in enumerate(arr) if val == 1.0] for arr in predicted_idx_list]
        # print(context_idxs)
        context_list=[]
        for sublist in context_idxs:
            sub_list=[]
            for j in sublist:
                sub_list.append(self.idx_to_word[j])
            context_list.append(sub_list)   
        for coreWord, predicted_words in zip(X_Test_context, context_list):
                print(f"中心词: '{coreWord}' -> 预测上下文: '{predicted_words}'")
        

origial_corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "Natural language processing is fascinating.",
    "Python makes text analysis easy and fun.",
    "Machine learning models need data to learn.",
    "This is a tiny corpus for testing purposes."
]
print("----------------------实现CBOW模型----------------")

# 使用CBOW模型根据上下文预测中间值
cobow = CBOWModel(origial_corpus, window_size=2)
X_context,y_center = cobow.create_corpus()
# 显示一些训练样本
print("\n---------训练样本示例:--------------")
for i in range(min(5, len(X_context))):
    center_word = cobow.idx_to_word[y_center[i]]
    print(f"  上下文: '{X_context[i]}' -> 中心词: '{center_word}'")

## 模型测试
print("\n ---------模型训练-----------")
cobow.train_and_fit(X_context,y_center)
print("\n ---------模型预测:-------------")
test_context=["quick brown", "need to learn"]
predicted_word=cobow.prediction(test_context)

print("\n----------------------实现Skip-gram模型----------------")



#使用Skip-gram模型根据上下文预测中间值
SkipGram = SkipGramModel(origial_corpus, window_size=2,embedding_dim=10)
X_center,y_context = SkipGram.create_corpus()
# 显示一些训练样本
print("\n---------训练样本示例:--------------")
for i in range(min(5, len(X_center))):
   context=[]
   y_out = [j for j, val in enumerate(y_context[i]) if val == 1]  
   for j in y_out:
        context.append(SkipGram.idx_to_word[j])
   print(f"中心词: '{X_center[i]}' -> 上下文: {context}")
 

print("\n ---------模型训练-----------")
SkipGram.train_and_fit(X_center,y_context)
print("\n ---------模型预测:-------------")
test_context=["Natural", "need"]
predicted_word=SkipGram.prediction(test_context)

