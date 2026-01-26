# import requests

# url="https://api.siliconflow.cn/v1/chat/completions"
# headers={
#     "Content-Type": "application/json",
#     "Authorization": "Bearer sk-cumwktcvhnxyqbqnhqjsoiknoozfklowfcdglkqvikohtjsl"
# }

# payload = {
#     "model": "Qwen/QwQ-32B",
#     "messages": [
#         {
#             "role": "user",
#             "content": "What opportunities and challenges will the Chinese large model industry face in 2025?"
#         }
#     ]
# }

# response = requests.post(url, headers=headers, json=payload)
# print(response.status_code)
# print(response.json())


# lst=[10,20,3,7]
# print(sum(lst))
# sum=0
# for i in range(len(lst)):
#     sum += lst[i]
# print(f"sum is {sum}")


# def getValue(dict,keyStr):
#     if keyStr in dict.keys():
#         return dict[keyStr]
#     else:
#         return -1


# def getValue2(dict,keyStr):
#    dict.get(keyStr)
# dictory={"AI":1,"NLP":2}
# print(str(getValue(dictory,"AI")))


# l1=[1,2,3,4]
# l2=[5,6,7,8]

# dictory2={}
# sentence=['I','I','love','NLP','LLM','AI','AI']
# for word in  sentence:
#     if word in dictory2:
#         dictory2[word]+=1
#     else:
#         dictory2[word]=1
# print(dictory2)

# for word in  sentence:
#     dictory2[word]=dictory2.get(word,0)+1
# print(dictory2)


# str1="Hello World"
# str2 = str1.upper()
# lst_str = str2.split('')
# print(lst_str)
# help(lst_str.sort)

# nums=[2,7,11,15,1,8]
# target=9

# for i in range(len(nums)):
#     for j in range(i+1,len(nums)):
#         if nums[i]+nums[j]==target:
#             print([i,j])
# lst3=[[i,j] for i in range(len(nums)) for j in range(len(nums)) if i!=j and nums[i]+nums[j]==target]
# print(lst3)

# def twoSum(nums, target):
#     result = []
#     for i in range(len(nums)):
#         for j in range(i+1,len(nums)):
#             if nums[i]+nums[j]==target:
#                 result.append([i,j])
#     return result

# print(twoSum(nums, target))

# nums=[0,1,0,3,12]
# def moveZeroes(nums):
#     j=0
#     for i in range(len(nums)):
#         if nums[i]!=0:
#             nums[j]=nums[i]
#             j+=1
#     for k in range(j,len(nums)):
#         nums[k]=0
#     return nums
# print(moveZeroes(nums))


# import numpy as np

# a = np.array([[1,2,3],[4,5,6]])
# b = np.random.random([2,3])
# print(b)
# print(np.argmax(a))
# print(np.argmax(b))
# np.zeros((3,6))
# print(np.sum(a))
# print(np.sum(a,axis=0))
# print(np.sum(a,axis=1))


# import requests
# url='https://api.siliconflow.cn/v1/chat/completions'
# headers={
#     'Authorization': 'Bearer sk-cumwktcvhnxyqbqnhqjsoiknoozfklowfcdglkqvikohtjsl',
#     'Content-Type': 'application/json'
# }
# data ={
#   "model": "Qwen/QwQ-32B",
#   "messages": [
#     {
#       "role": "user",
#       "content": "What opportunities and challenges will the Chinese large model industry face in 2025?"
#     }
#   ]
# }
# response = requests.post(url,headers=headers,json=data)
# print (response.status_code)


# 1. 给定一个数值列表（向量），v = [10, 20, 3, 7]，请计算并返回列表中所有元素的总和。
# lst=[10, 20, 3, 7]
# print(sum(lst))
# sum = 0
# for i in lst:
#     sum += i
# print(f"sum is {sum}")
# print(dir(lst))
# help(lst.pop)

# print(lst)
# 2. 给定一个词汇表 vocab 字典 vocab = {'AI': 1, 'NLP': 2} ，请实现一个函数，输入单词，返回其对应的 ID。如果单词不在词汇表中，返回 -1。
dictory = {"AI": 1, "NLP": 2}
# word = input("请输入一个单词")
# if word in dictory:
#     print(dictory[word])
# else:
#     print(-1)
# print(dictory.get(word,-1))
# print(dir(dictory))
# help(dictory.update)
# dictory.update({'ML':3})
# print(dictory)
# dictory.update({'NLP':10})
# print(dictory)


# # 3. 给定两个相同长度的数值列表（向量）v1 = [1, 5, 9], v2 = [2, 3, 1]，请实现对应元素相加操作，并返回一个新的结果列表
# v1 = [1, 5, 9]
# v2 = [2, 3, 1]
# v3 = list()
# for i in range(len(v1)):
#     v3.append(v1[i] + v2[i])
# print(v3)

# v3 = [v1[i] + v2[i] for i in range(len(v1))]
# print(v3)

# 4. 给定一个已分词的句子列表 sentence = ['我', '爱', '学习', '我', '爱', 'NLP']，请统计每个词出现的次数，并以字典形式返回。
# dictory2=dict()
# sentence=['I','I','love','NLP','LLM','AI','AI']
# # for word in sentence:
# #     dictory2[word]=dictory2.get(word,0)+1

# for word in sentence:
#     if word in dictory2:
#        dictory2[word]+=1
#     else:
#         dictory2[word]=1
# print(dictory2)

# 5. 给定一个句子 text = "NLP Is Awesome" ，请将句子中的所有字母转换为小写，并将句子按照空格切分成一个列表（分词）。
# text = "NLP Is Awesome"
# text.lower().split(' ')

"""
给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案，并且你不能使用两次相同的元素。

你可以按任意顺序返回答案。

"""
# nums=[2,7,11,15,1,8]
# target=9
# for i in range(len(nums)):
#     for j in range(i+1,len(nums)):
#         if nums[i]+nums[j]==target:
#             print([i,j])

# dictory4 ={}
# i=0;
# for item in nums:
#     dictory4[item]=i
#     i+=1
# print(dictory4)
# for i in range(len(nums)):
#     tareget_num=target - nums[i]
#     #print(tareget_num)
#     if tareget_num in dictory4 :
#         print([i,dictory4.get(tareget_num)])


##给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。

##请注意 ，必须在不复制数组的情况下原地对数组进行操作。

# nums=[0,1,0,3,12]
# j=0
# for i in range(len(nums)):
#     if nums[i]!=0:
#         nums[j]=nums[i]
#         j+=1
# for k in range(j,len(nums)):
#     nums[k]=0
# print(nums)


# from collections import defaultdict
# from collections import Counter

# nums5=[0,1,0,1,12]
# print(Counter(nums5))

# default_dict=defaultdict(int)
# for item in nums5:
#     default_dict[item] += 1
# print (default_dict)

# default_dict2=defaultdict(list)
# for item in nums5:
#     default_dict2[item]=default_dict2.get(item,0)+1
# print (default_dict2)

# import pandas as pd
# import numpy as np

# a = np.random.normal(5,0.2,5)
# # print(a)
# # print(np.var(a))
# # print(np.mean(a))
# b=np.random.randn(10,3)
# print(b)


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

# plt.rcParams["font.sans-serif"] = ["KaiTi"]
# plt.rcParams["axes.unicode_minus"] = False

# def polynomial(x, degree):
#     """构成多项式，返回 [x^1,x^2,x^3,...,x^n]"""
#     return np.hstack([x**i for i in range(1, degree + 1)])

# # 生成随机数据
# X = np.linspace(-3, 3, 300).reshape(-1, 1)
# y = np.sin(X) + np.random.uniform(-0.5, 0.5, 300).reshape(-1, 1)
# fig, ax = plt.subplots(1, 3, figsize=(15, 4))
# ax[0].plot(X, y, "yo")
# ax[1].plot(X, y, "yo")
# ax[2].plot(X, y, "yo")

# # 划分训练集和测试集
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 创建线性回归模型
# model = LinearRegression()

# # 欠拟合
# x_train1 = x_train
# x_test1 = x_test
# model.fit(x_train1, y_train)  # 模型训练
# y_pred1 = model.predict(x_test1)  # 预测
# ax[0].plot(np.array([[-3], [3]]), model.predict(np.array([[-3], [3]])), "c")  # 绘制曲线
# ax[0].text(-3, 1, f"测试集均方误差：{mean_squared_error(y_test, y_pred1):.4f}")
# ax[0].text(-3, 1.3, f"训练集均方误差：{mean_squared_error(y_train, model.predict(x_train1)):.4f}")

# # 恰好拟合
# x_train2 = polynomial(x_train, 5)
# x_test2 = polynomial(x_test, 5)
# print(f'x_train.shape: {x_train.shape}')  ##(240, 1)
# print(f'x_train: {x_train}')
# print(f'x_train2.shape: {x_train2.shape}')  ##(240, 5)
# print(f'x_train2: {x_train2}')
# # model.fit(x_train2, y_train)  # 模型训练
# # y_pred2 = model.predict(x_test2)  # 预测
# # ax[1].plot(X, model.predict(polynomial(X, 5)), "k")  # 绘制曲线
# # ax[1].text(-3, 1, f"测试集均方误差：{mean_squared_error(y_test, y_pred2):.4f}")
# # ax[1].text(-3, 1.3, f"训练集均方误差：{mean_squared_error(y_train, model.predict(x_train2)):.4f}")


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression, Lasso, Ridge
# from sklearn.metrics import mean_squared_error

# plt.rcParams["font.sans-serif"] = ["KaiTi"]
# plt.rcParams["axes.unicode_minus"] = False

# def polynomial(x, degree):
#     """构成多项式，返回 [x^1,x^2,x^3,...,x^n]"""
#     return np.hstack([x**i for i in range(1, degree + 1)])

# # 生成随机数据
# X = np.linspace(-3, 3, 300).reshape(-1, 1)
# y = np.sin(X) + np.random.uniform(-0.5, 0.5, X.size).reshape(-1, 1)
# fig, ax = plt.subplots(2, 3, figsize=(15, 8))
# ax[0, 0].plot(X, y, "yo")
# ax[0, 1].plot(X, y, "yo")
# ax[0, 2].plot(X, y, "yo")

# # 划分训练集和测试集
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# x_train1 = polynomial(x_train, 20)
# x_test1 = polynomial(x_test, 20)

# # 拟合
# model = LinearRegression()
# model.fit(x_train1, y_train)  # 模型训练
# y_pred3 = model.predict(x_test1)  # 预测
# ax[0, 0].plot(X, model.predict(polynomial(X, 20)), "r")  # 绘制曲线
# ax[0, 0].text(-3, 1, f"测试集均方误差：{mean_squared_error(y_test, y_pred3):.4f}")
# ax[1, 0].bar(np.arange(20), model.coef_.reshape(-1))  # 绘制所有系数

# # L1正则化-Lasso回归
# lasso = Lasso(alpha=0.01)
# lasso.fit(x_train1, y_train)  # 模型训练
# y_pred3 = lasso.predict(x_test1)  # 预测
# ax[0, 1].plot(X, lasso.predict(polynomial(X, 20)), "r")  # 绘制曲线
# ax[0, 1].text(-3, 1, f"测试集均方误差：{mean_squared_error(y_test, y_pred3):.4f}")
# ax[0, 1].text(-3, 1.2, "Lasso回归")
# ax[1, 1].bar(np.arange(20), lasso.coef_)  # 绘制所有系数

# # L2正则化-岭回归
# ridge = Ridge(alpha=1)
# ridge.fit(x_train1, y_train)  # 模型训练
# y_pred3 = ridge.predict(x_test1)  # 预测
# ax[0, 2].plot(X, ridge.predict(polynomial(X, 20)), "r")  # 绘制曲线
# ax[0, 2].text(-3, 1, f"测试集均方误差：{mean_squared_error(y_test, y_pred3):.4f}")
# ax[0, 2].text(-3, 1.2, "岭回归")
# ax[1, 2].bar(np.arange(20), ridge.coef_)  # 绘制所有系数
# plt.show()


import numpy as np

# def softmax(x):
#     """Softmax函数，将数值转换为概率分布"""
#     exp_x = np.exp(x - np.max(x))  # 减去最大值防止数值溢出
#     return exp_x / np.sum(exp_x)

# # 示例语料
# corpus = "deep learning is very deep and learning is fun"
# tokens = corpus.split()
# vocab = set(tokens)
# vocab_size = len(vocab)
# word_to_ix = {word: i for i, word in enumerate(vocab)}
# ix_to_word = {i: word for word, i in word_to_ix.items()}
# print('word_to_ix:',word_to_ix)
# cbow_data = []
# for i in range(2, len(tokens) - 2):
#     # 上下文窗口: [i-2, i-1, i+1, i+2]
#     context = [tokens[i-2], tokens[i-1], tokens[i+1], tokens[i+2]]
#     target = tokens[i]  # 中心词
#     # print("-----------------------")
#     # print("context:",context)
#     # print("target:",target)

#     # 转换为索引
#     context_idxs = [word_to_ix[w] for w in context]
#     target_idx = word_to_ix[target]

#     cbow_data.append((context_idxs, target_idx))
#     print(cbow_data)

# for i, (ctx, tgt) in enumerate(cbow_data):
#     # print("ctx:",ctx)
#     # print("tgt:",tgt)
#     # print('word_to_ix',word_to_ix)
#     print(f"  样本 {i+1}: 上下文 {[ix_to_word[w] for w in ctx]} -> 中心词 '{ix_to_word[tgt]}'")

# context_idxs=[3, 2, 4, 3]
# W1 = np.random.randn(6,10)
# x = W1[context_idxs]
# print(x)

# np.random.choice、

# x=np.array([-10,20,30,-40,50])
# y=x>0
# print(y)
# print(x[y])

# import matplotlib.pyplot as plt
# import numpy as np


# def softmax(x):
#     """Softmax函数，将数值转换为概率分布"""
#     exp_x = np.exp(x - np.max(x))  # 减去最大值防止数值溢出
#     return exp_x / np.sum(exp_x)

# # 示例语料
# corpus = "deep learning is very deep and learning is fun"
# tokens = corpus.split()
# vocab = set(tokens)
# vocab_size = len(vocab)
# embed_size = 10  # 词向量维度
# epochs = 100
# learning_rate = 0.01

# # 创建词汇表映射
# word_to_ix = {word: i for i, word in enumerate(vocab)}
# ix_to_word = {i: word for word, i in word_to_ix.items()}

# W1 = np.random.randn(vocab_size, embed_size) * 0.01  # 使用小随机数初始化

# W2 = np.random.randn(embed_size, vocab_size) * 0.01

# cbow_data = []
# for i in range(2, len(tokens) - 2):
#     # 上下文窗口: [i-2, i-1, i+1, i+2]
#     context = [tokens[i-2], tokens[i-1], tokens[i+1], tokens[i+2]]
#     target = tokens[i]  # 中心词

#     # 转换为索引
#     context_idxs = [word_to_ix[w] for w in context]
#     target_idx = word_to_ix[target]

#     cbow_data.append((context_idxs, target_idx))
# print("\n--- 开始训练 CBOW (NumPy) ---")

# import tensorflow as tf
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# (x_train, y_train), (x_test, y_test) =tf.keras.datasets.mnist.load_data()
# print(x_train.shape,y_train.shape)
# x_train= x_train.reshape(-1,28*28)
# x_test= x_test.reshape(-1,28*28)
# x_test = x_test / 255.0
# x_train= x_train/255

# model = LogisticRegression()
# model.fit(x_train, y_train)
# y_pred= model.predict(x_test)
# accuracy_score(y_test,y_pred)

# import torch
# # tensor = torch.tensor([[1, 2], [3, 4]])
# # print(tensor)
# # print(tensor.dtype)
# tensor1 = torch.logspace(1, 3, 1, 2)
# print(tensor1)
# torch.manual_seed(42)
# print(torch.rand(2,3))

# import torch
# # tensor1 = torch.randint(1, 9, (3, 5, 4))
# # print(tensor1)
# # # print(tensor1[:, :, 0] > 5)
# # # print(tensor1[tensor1[:, :, 0] > 5])

# # mask = tensor1[:, 1, :] > 5
# # print(mask)
# # print(tensor1[tensor1[:, 1, :] > 5])
# tensor1 = torch.randint(1, 9, (3, 5,4))
# print(tensor1)
# print(tensor1.transpose(0,2))
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# import numpy as np
# X = [[5], [8], [10], [12], [15], [3], [7], [9], [14], [6]]
# # 因变量，数学考试成绩
# y = [55, 65, 70, 75, 85, 50, 60, 72, 80, 58]
# # 实例化线性回归模型
# model = LinearRegression()
# # 模型训练
# model.fit(X, y)
# X_new=np.arange(0,15,0.1).reshape(-1,1)
# y_pred=model.predict(X_new)
# plt.scatter(X,y)
# plt.plot(X_new,y_pred,'b')
# plt.show()

# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.linear_model import LogisticRegression
# import matplotlib.pyplot as plt
# digits = datasets.load_digits()
# image_data=digits.images
# label_data=digits.target
# print(image_data.shape)
# print(label_data.shape)
# X_train, X_test, y_train, y_test=train_test_split(image_data,label_data,test_size=0.2,random_state=42)
# # 归一化
# print(X_train.shape)
# print(X_test.shape)
# X_train=X_train.reshape(1437,64)
# X_test=X_test.reshape(360,64)
# preprocessor = MinMaxScaler()
# x_train = preprocessor.fit_transform(X_train)
# x_test = preprocessor.transform(X_test)

# # # 模型训练
# model = LogisticRegression(max_iter=500)
# model.fit(x_train, y_train)

# # # 模型评估
# model.score(x_test, y_test)

# # # 预测
# plt.imshow(x_test[10].reshape(8, 8), cmap="gray")
# plt.show()
# print(model.predict(x_test[10].reshape(1, -1)))

import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

# x = torch.tensor(10.0)
# # 目标值y
# y = torch.tensor(3.0)

# # 初始化权重w
# w = torch.rand(1, 1, requires_grad=True)
# # 初始化偏置b
# b = torch.rand(1, 1, requires_grad=True)
# z = w * x + b
# print(y.is_leaf)


import torch
import torch.nn as nn

# import jieba

# # 设置随机种子
# torch.manual_seed(42)
# text = "自然语言是由文字构成的，而语言的含义是由单词构成的。即单词是含义的最小单位。因此为了让计算机理解自然语言，首先要让它理解单词含义。"
# # 自定义停用词和标点符号
# stopwords = {"的", "是", "而", "由", "，", "。", "、"}
# # 分词，过滤停用词和标点，去重，构建词表
# words = [word for word in jieba.lcut(text) if word not in stopwords]
# vocab = list(set(words))  # 词表
# # 构建词到索引的映射
# word2idx = dict()
# for idx, word in enumerate(vocab):
#     word2idx[word] = idx
# 初始化嵌入层
# embed = nn.Embedding(num_embeddings=len(word2idx), embedding_dim=5)
# print(embed)
# # # 打印词向量
# for idx, word in enumerate(vocab):
#     word_vec = embed(torch.tensor(idx))  # 通过索引获取词向量
#     print(f"{idx:>2}:{word:8}\t{word_vec.detach().numpy()}")


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from collections import Counter, deque
# import numpy as np
# import pickle

# # ----------------------------
# # 1. 超参数设置
# # ----------------------------
# CONTEXT_SIZE = 2        # 上下文窗口大小（左右各2个词）
# EMBEDDING_DIM = 100     # 词向量维度
# EPOCHS = 100
# LR = 0.01
# BATCH_SIZE = 64

# # 示例语料（可替换为更大语料）
# corpus = """
# We are about to study the idea of a computational process.
# Computational processes are abstract beings that inhabit computers.
# As they evolve, processes manipulate other abstract things called data.
# The evolution of a process is directed by a pattern of rules called a program.
# People create programs to direct processes.
# In effect, we conjure the spirits of the computer with our spells.
# """.strip().split()

# # ----------------------------
# # 2. 构建词汇表
# # ----------------------------
# word_counts = Counter(corpus)
# vocab = sorted(word_counts, key=word_counts.get, reverse=True)
# vocab_size = len(vocab)

# word_to_idx = {word: i for i, word in enumerate(vocab)}
# idx_to_word = {i: word for i, word in enumerate(vocab)}

# print(f"Vocabulary size: {vocab_size}")
# print("Sample vocab:", vocab[:10])

# # ----------------------------
# # 3. 生成 CBOW 训练数据
# # ----------------------------
# def create_cbow_dataset(corpus, context_size):
#     data = []
#     for i in range(context_size, len(corpus) - context_size):
#         context = []
#         # 左右上下文
#         for j in range(-context_size, context_size + 1):
#             if j == 0:
#                 continue
#             context.append(corpus[i + j])
#         target = corpus[i]
#         data.append((context, target))
#     return data

# dataset = create_cbow_dataset(corpus, CONTEXT_SIZE)
# # print(dataset)
# print(f"\nNumber of training samples: {len(dataset)}")
# print("Example sample:", dataset[0])

# # 转换为索引
# def preprocess(data):
#     X, y = [], []
#     for context, target in data:
#         ctx_indices = [word_to_idx[w] for w in context]
#         tgt_index = word_to_idx[target]
#         X.append(ctx_indices)
#         y.append(tgt_index)
#     return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# X, y = preprocess(dataset)
# print(X)
# print(y)
# class CBOW(nn.Module):
#     def __init__(self, vocab_size, embedding_dim):
#         super(CBOW, self).__init__()
#         self.embeddings = nn.Embedding(vocab_size, embedding_dim)
#         self.linear = nn.Linear(embedding_dim, vocab_size)

#     def forward(self, context_idxs):
#         # context_idxs: [batch_size, 2 * CONTEXT_SIZE]
#         embeds = self.embeddings(context_idxs)  # [B, 2C, D]
#         # 对上下文向量求平均
#         avg_embeds = torch.mean(embeds, dim=1)  # [B, D]
#         logits = self.linear(avg_embeds)        # [B, V]
#         return logits

# # ----------------------------
# # 5. 训练
# # ----------------------------
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = CBOW(vocab_size, EMBEDDING_DIM).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=LR)

# # 创建 DataLoader（简单批处理）
# from torch.utils.data import TensorDataset, DataLoader
# train_dataset = TensorDataset(X, y)
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# print("\nStart training...")
# for epoch in range(EPOCHS):
#     total_loss = 0
#     for batch_X, batch_y in train_loader:
#         batch_X, batch_y = batch_X.to(device), batch_y.to(device)

#         optimizer.zero_grad()
#         logits = model(batch_X)
#         loss = criterion(logits, batch_y)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     if (epoch + 1) % 20 == 0:
#         print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

# ----------------------------
# 6. 提取并保存词向量
# ----------------------------
# # 获取 Embedding 层权重（即词向量）
# embedding_matrix = model.embeddings.weight.data.cpu().numpy()  # [vocab_size, EMBEDDING_DIM]

# # 保存词向量（两种方式）
# # 方式1：保存为 .npy 文件（纯向量）
# np.save('cbow_word_vectors.npy', embedding_matrix)

# # 方式2：保存为字典 {word: vector}
# word_vectors = {}
# for word, idx in word_to_idx.items():
#     word_vectors[word] = embedding_matrix[idx]

# with open('cbow_word_vectors.pkl', 'wb') as f:
#     pickle.dump(word_vectors, f)

# # 方式3：保存整个模型（可用于继续训练或推理）
# torch.save(model.state_dict(), 'cbow_model.pth')

# # 同时保存词汇表映射（重要！）
# vocab_info = {
#     'word_to_idx': word_to_idx,
#     'idx_to_word': idx_to_word,
#     'vocab': vocab
# }
# with open('vocab_info.pkl', 'wb') as f:
#     pickle.dump(vocab_info, f)

# print("\n✅ 模型和词向量已保存！")
# print("文件列表:")
# print("- cbow_word_vectors.npy   # 词向量矩阵")
# print("- cbow_word_vectors.pkl   # 词到向量的字典")
# print("- cbow_model.pth          # PyTorch 模型参数")
# print("- vocab_info.pkl          # 词汇表信息")


from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="./logs")
writer.add_scalar()