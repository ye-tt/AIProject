import numpy as np
np.set_printoptions(precision=2)
#scikit-learn 主要支持传统的机器学习算法，不适用于深度学习（如神经网络的复杂架构）。深度学习通常使用 TensorFlow、PyTorch 等框架。
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from lab_utils_multi import  load_house_data
import matplotlib.pyplot as plt
dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0'; 
plt.style.use('./deeplearning.mplstyle')

'''
#################single variable linear regression example####################
X_train = np.array([1.0, 2.0])   #features
y_train = np.array([300, 500])   #target value

#创建线性回归对象
linear_model = LinearRegression()

#X must be a 2-D Matrix
#utilizes one of the methods associated with the object, `fit`. This performs regression, fitting the parameters to the input data
linear_model.fit(X_train.reshape(-1, 1), y_train) 
#w b parameters are referred to as 'coefficients'(系数) and 'intercept'（截距） in scikit-learn.
#eg: 在线性回归 y = w₁x₁ + w₂x₂ + b 中，w₁ 和 w₂ 就是特征 x₁ 和 x₂ 的系数（coefficients），b 是截距（intercept）
b = linear_model.intercept_
w = linear_model.coef_
print(f"w = {w:}, b = {b:0.2f}")
print(f"'manual' prediction: f_wb = wx+b : {1200*w + b}")


y_pred = linear_model.predict(X_train.reshape(-1, 1))

print("Prediction on training set:", y_pred)

X_test = np.array([[1200]])
print(f"Prediction for 1200 sqft house: ${linear_model.predict(X_test)[0]:0.2f}")
'''


#################multiple features linear regression example####################

# load the dataset
X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']
linear_model = LinearRegression()
linear_model.fit(X_train, y_train) 
b = linear_model.intercept_
w = linear_model.coef_
print(f"w = {w:}, b = {b:0.2f}")
print(f"Prediction on training set:\n {linear_model.predict(X_train)[:4]}" )
print(f"prediction using w,b:\n {(X_train @ w + b)[:4]}")
print(f"Target values \n {y_train[:4]}")

x_house = np.array([1200, 3,1, 40]).reshape(-1,4)
x_house_predict = linear_model.predict(x_house)[0]
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.2f}")