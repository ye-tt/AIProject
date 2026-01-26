import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import OneHotEncoder




(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train= x_train.reshape(-1,28*28)
x_test= x_test.reshape(-1,28*28)
# x(6000,784) y(6000,)
# print(x_train.shape,y_train.shape)
x_train = x_train / 255.0
x_test = x_test / 255.0

print('=================sklearn 实现===================')
#sklearn 实现
modle = LogisticRegression(max_iter=1000)
modle.fit(x_train, y_train)

y_pred = modle.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


#  基于Numpy 实现

def softmax(x):
    #x 形状是 (batch_size, 10)，axis=1, 按行求最大值，将上一层输出的任意实数值（Logits）转换为概率分布，确保所有类别的预测概率之和为 1
    x = x - np.max(x, axis=1,keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=1,keepdims=True)
'''
X(1,784) W(784,10) b(1,10)
'''
def preprocess(x_train, y_train):
    encoder = OneHotEncoder(sparse_output=False)
    y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
    model_numpy = LogisticRegressionNumpy(learning_rate=00.1)
    data_indexs = np.arange(x_train.shape[0])
    np.random.shuffle(data_indexs)
    x_train = x_train[data_indexs]
    y_train_onehot = y_train_onehot[data_indexs]
    return x_train, y_train_onehot
# numpy 实现
class LogisticRegressionNumpy:
    def __init__(self, learning_rate=0.01):
        self.W = np.random.randn(784, 10) * 0.01
        self.b = np.zeros((1, 10))
        self.learning_rate = learning_rate
    
    def forword(self,X):
        Z =np.dot(X,self.W) + self.b
        y_pred = softmax(Z)
        return y_pred
    
    def loss(self,y_true,y_pred):
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        return loss
    
    def backword(self,X,y_true,y_pred):
        m = X.shape[0]
        dL= y_pred - y_true
        dw = np.dot(X.T, dL) / m
        #keepdims=True：在执行降维操作（如求和、求均值）后，是否保留被压缩的维度（使其长度为 1）
        db = np.sum(dL, axis=0, keepdims=True) /m 
        return dw, db
    
    def update_parameters(self, dw, db):
        self.W -= self.learning_rate * dw
        self.b -= self.learning_rate * db

print('=================Numpy 实现===================')
x_train, y_train_onehot = preprocess(x_train, y_train)
model_numpy = LogisticRegressionNumpy(learning_rate=00.1)
epochs = 50
batch_size = 100

for epoch in range(epochs):
    total_loss=0
    itr_num= np.ceil(x_train.shape[0]/batch_size).astype(int)
    for itr in range(itr_num):
        start = itr * batch_size
        end = start + batch_size
        x_batch = x_train[start:end]
        y_batch= y_train_onehot[start:end]

        y_batch_pred=model_numpy.forword(x_batch)
        # print("22222222222",y_batch_pred.shape)
        loss = model_numpy.loss(y_batch, y_batch_pred)
        total_loss += loss
        dw, db = model_numpy.backword(x_batch, y_batch, y_batch_pred)
        model_numpy.update_parameters(dw,db)
    if (epoch + 1) % 10 == 0:
        train_pred = model_numpy.forword(x_train[:1000])  # 用部分训练集快速评估
        test_pred = model_numpy.forword(x_test)
        train_pred_result = np.argmax(train_pred, axis=1)
        test_pred_result = np.argmax(test_pred, axis=1)
        # print('train_pred:',train_pred.shape)
        # print('test_pred:',test_pred.shape)
        # print('test_pred:',test_pred.shape)
        train_accuracy = np.mean(train_pred_result == y_train[:1000])
        test_acc_accuracy = np.mean(test_pred_result == y_test)
        avg_loss = total_loss / itr_num
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_acc_accuracy:.4f}")
