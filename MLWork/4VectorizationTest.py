import numpy as np
import time

##These have all created a one-dimensional vector
# NumPy routines which allocate memory and fill arrays with value
# a = np.zeros(4);                print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
# a = np.zeros((4,));             print(f"np.zeros(4,) :  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
# a = np.random.random_sample(4); print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
# # NumPy routines which allocate memory and fill arrays with value but do not accept shape as input argument
# a = np.arange(4.);              print(f"np.arange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
# a = np.random.rand(4);          print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

#####
# a = np.zeros((4,2));             print(f"np.zeros(4,2) :  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

a = np.array([1,2,3,4])
#print(f"a             : {a}")
# negate elements of a
b = -a 
#print(f"b = -a        : {b}")
# sum all elements of a, returns a scalar
b = np.sum(a) 
#print(f"b = np.sum(a) : {b}")
b = np.mean(a)
#print(f"b = np.mean(a): {b}")
b = a**2
#print(f"b = a**2      : {b}")
def my_dot(a, b): 
    """
   Compute the dot product of two vectors
 
    Args:
      a (ndarray (n,)):  input vector 
      b (ndarray (n,)):  input vector with same dimension as a
    
    Returns:
      x (scalar): 
    """
    x=0
    for i in range(a.shape[0]):
        x = x + a[i] * b[i]
    return x

a = np.array([1,2,3,4])
b = np.array([-1,4,3,2])
# print(f"my_dot(a,b)={my_dot(a, b)}")
# print(f"np.dot(a,b)={np.dot(a, b)}")


'''
np.random.seed(1)
a = np.random.rand(10000000)  # very large arrays
b = np.random.rand(10000000)

tic = time.time()  # capture start time
c = np.dot(a, b)
toc = time.time()  # capture end time

print(f"np.dot(a, b) =  {c:.4f}")
print(f"Vectorized version duration: {1000*(toc-tic):.4f} ms ")

tic = time.time()  # capture start time
c = my_dot(a,b)
toc = time.time()  # capture end time

print(f"my_dot(a, b) =  {c:.4f}")
print(f"loop version duration: {1000*(toc-tic):.4f} ms ")

del(a);del(b)  #remove these big arrays from memory
'''



# show common Course 1 example
X = np.array([[1],[2],[3],[4]])
w = np.array([2])
c = np.dot(X[1], w)

# print(f"X[1] has shape {X[1].shape}")
# print(f"w has shape {w.shape}")
# print(f"c has shape {c.shape}")

a = np.zeros((1, 5))                                       
# print(f"a = np.zeros((1, 5)) shape = {a.shape}, a = {a}")                     

a = np.zeros((2, 1))                                                                   
# print(f"a = np.zeros((2, 1)) shape = {a.shape}, a = {a}") 

a = np.random.random_sample((1, 1))  
# print(f"a = np.random.random_sample((1, 1)) shape = {a.shape}, a = {a}") 


a = np.array([[5], [4], [3]]);   
# print(f" a np.array([[5], [4], [3]]) shape = {a.shape}, np.array: a = {a}")
a = np.array([[5],   # One can also
              [4],   # separate values
              [3]]); #into separate rows
# print(f" a shape = {a.shape}, np.array: a = {a}")


#vector indexing operations on matrices
a = np.arange(6).reshape(-1, 2)   #reshape is a convenient way to create matrices
print(f"a.shape: {a.shape}, \na= {a}")
# 上面的代码等同于
a1 = np.arange(6)  #创建一维数组[0,1,2,3,4,5]
a2 = a.reshape(3, 2) #变成3行2列

#access an element
print(f"\na[2,0].shape:   {a[2, 0].shape}, a[2,0] = {a[2, 0]},     type(a[2,0]) = {type(a[2, 0])} Accessing an element returns a scalar\n")

#access a row
print(f"a[2].shape:   {a[2].shape}, a[2]   = {a[2]}, type(a[2])   = {type(a[2])}")



#vector 2-D slicing operations
a = np.arange(20).reshape(2, 10)
print(f"a = \n{a}")

#access 5 consecutive elements (start:stop:step)
print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")

#access 5 consecutive elements (start:stop:step) in two rows
print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")

# access all elements
print("a[:,:] = \n", a[:,:], ",  a[:,:].shape =", a[:,:].shape)

# access all elements in one row (very common usage)
print("a[1,:] = ", a[1,:], ",  a[1,:].shape =", a[1,:].shape, "a 1-D array")
# same as
print("a[1]   = ", a[1],   ",  a[1].shape   =", a[1].shape, "a 1-D array")
