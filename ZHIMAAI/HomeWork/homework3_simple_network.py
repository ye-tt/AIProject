## 定义网络
'''
反向传误差 = 链式法则+激活函数导数
权重相关的传递：始终是乘上一层权重的转置
激活函数相关的传递：始终是乘当前层激活函数的导数
'''
import numpy as np

## 1*4 @ 4*2 -> 1*2
class SimpleNet:
    def __init__(self, layer_sizes, learning_rate=0.1):

        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.parameters = {}
        self.initialize_parameters()
    
    def initialize_parameters(self):
        """初始化权重和偏置"""
        np.random.seed(42)
        ##初始化 W1, b1, W2, b2, 
        for l in range(1, len(self.layer_sizes)):
            self.parameters[f'W{l}'] = np.random.randn(
                self.layer_sizes[l], self.layer_sizes[l-1])
            print(f"self.parameters[f'W{l}'].shape: {self.parameters[f'W{l}'].shape}")
            self.parameters[f'b{l}'] = np.zeros((self.layer_sizes[l], 1))  
            print(f"self.parameters[f'b{l}'].shape: {self.parameters[f'b{l}'].shape}")   

    
    def relu(self, Z):
        """ReLU激活函数"""
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        """ReLU导数"""
        return (Z > 0).astype(float)
    
    def softmax(self, Z):
        """Softmax激活函数"""
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # 防止数值溢出
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    

    def forward_propagation(self, X):
        """前向传播"""
        cache = {'A0': X}
        # 隐藏层: 线性变换 + ReLU
        # 8*4 @ 4*1 +8*1 ->8*1
        cache[f'Z1'] = np.dot(self.parameters[f'W1'], cache[f'A0']) + self.parameters[f'b1']
        cache[f'A1'] = self.relu(cache[f'Z1'])
        
        # A1 (8,1)
        # 输出层: 线性变换 +softmax
        # 2*8 @ 8*1 + 2*1 ->  2*1      
        cache[f'Z2'] = np.dot(self.parameters[f'W2'], cache[f'A1']) + self.parameters[f'b2']
        print(f"cache[f'Z2'] is: {cache[f'Z2']}")
        cache[f'A2'] = self.softmax(cache[f'Z2'])        
        return cache
    
    def compute_loss(self, AL, Y):
        """计算交叉熵损失"""
        m = Y.shape[1]
        # 添加小值防止log(0)
        #参考逻辑回归cost 函数
        loss = -np.sum(Y * np.log(AL + 1e-8)) / m
        return loss
    
    def backward_propagation(self, X, Y, cache):
        """反向传播"""
        grads = {}
        # 输出层梯度
        # A2 (2,1), Y(1,2)

        '''
        反向传误差 = 链式法则+激活函数导数
        权重相关的传递：始终是乘上一层权重的转置 W{i+1}.T
        激活函数相关的传递：始终是乘当前层激活函数的导数
        '''
        #计算输出层误差
        dZ2 = cache[f'A2'] - Y.T  ###loss 函数对Z2 的偏导数dL/dZ2 其中Z2=W2@A1+b2
        print(f'dZ2={dZ2}') 
        print(f"cache[f'A2'].shape={cache[f'A2'].shape}")  
        # 更新 W2 和 b2
        grads[f'dW2'] = np.dot(dZ2, cache[f'A1'].T) 
        grads[f'db2'] = np.sum(dZ2, axis=1, keepdims=True)   
        
        # 反向传播到隐藏层
        dA = np.dot(self.parameters[f'W{2}'].T, dZ2) 
        dZ1 = dA * self.relu_derivative(cache[f'Z1'])
        # 更新 W1 和 b1
        grads[f'dW1'] = np.dot(dZ1, cache[f'A0'].T)
        grads[f'db1'] = np.sum(dZ1, axis=1, keepdims=True)     
        return grads
    
    def update_parameters(self, grads):
        """更新参数"""     
        for l in range(1, 3):
            self.parameters[f'W{l}'] -= self.learning_rate * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= self.learning_rate * grads[f'db{l}']


'''
输入层: (1, 2, 2) → reshape (1, 4) 1*2*2 ->1*4 
隐藏层 W1: 4*8, b1: 1*8
1*4 @ 4*8 -> 1*8
使用Relu 激活函数
输出层 W2: 8*2, b2: 1*2
1*8 @ 8*2 -> 1*2
用softmax激活函数 (二分类)
'''

layer_sizes=[4,8,2]
learning_rate=0.1
X = np.random.rand(1, 2, 2)  # 输入
X = X.reshape(1,4)
X = X.T
print("Input X:\n", X)
Y_true = np.array([[1.0, 0.0]])  # 假设真实标签（用于计算 loss）    
net = SimpleNet(layer_sizes, learning_rate)


# 前向
cache = net.forward_propagation(X) 
# print("Output cache:\n", cache)
# print("\n")
Y_pred=cache[f'A2']
print('Y_pred：',Y_pred)
loss = net.compute_loss(Y_pred,Y_true.T)
print(f'Initial loss: {loss}')
grads = net.backward_propagation(X, Y_true, cache)
# print(f"Gradients:{grads}")
net.update_parameters(grads)

# 再次前向传播查看更新后的损失
print('X is :',X)
cache_new = net.forward_propagation(X)
Y_pred_new=cache_new[f'A2']
loss_new = net.compute_loss(Y_pred_new, Y_true.T)
print("Loss after one step:", loss_new)