import numpy as np

def transform(X,layer_sizes):
    for i in range(1,len(layer_sizes)):
        print(i)
        W=np.random.randn(layer_sizes[i-1],layer_sizes[i])
        print(f'X.shape:{X.shape}')
        print(f'W.shape:{W.shape}')        
        A_out = X @ W
        X = A_out
    return A_out


img = np.random.randn(1,32, 32)
img= img.reshape(1,1024)

#中间过程经过1次矩阵变换 1*32*32 1*1024 @ 1024*10 -> 1*10
layer_sizes=[1024,10]
output1 = transform(img,layer_sizes)
print(f'output1.shape:{output1.shape}')

'''
中间过程经过3次矩阵变换 1*1024 @ 1024*256 -> 1*256 @ 256*64
-> 1*64 @ 64 * 10 -> 1*10
'''
layer_sizes3=[1024,256,64,10]
output3 = transform(img,layer_sizes3)
print(f'output3.shape:{output3.shape}')

#中间过程经过5次矩阵变换
layer_sizes5=[1024,512,256,128,64,10]
output5= transform(img,layer_sizes5)
print(f'output5.shape:{output5.shape}')
