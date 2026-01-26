import requests
from bs4 import BeautifulSoup

import multiprocessing as mp
import  threading
from time import sleep
import numpy as np
import matplotlib.pyplot as plt

# def fetch_page_title(url):
#     header={
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
#     }
#     response = requests.get(url,headers=header)
#     response.encoding = 'utf-8'
#     soup = BeautifulSoup(response.text, 'html.parser')
#     return soup.title.getText()

# url = "https://www.google.com"
# title = fetch_page_title(url)
# print(f'The title of the page is: {title}')
# threading.Thread()
# sleep(1) 

plt.style.use('deeplearning.mplstyle')
x_train= np.array([1.0,2.0])
y_train= np.array([300.0,500.0])
print(f"x_train={x_train}, y_train={y_train}")
print(f"x_train shape={x_train.shape}")
m= x_train.shape[0]
print(f"Number of training examples: m={m}")
# Or we can use len to get the number of training examples
# m=len(x_train)
# print(f"Number of training examples using len: {m}")

i=0
x_i=x_train[i]
y_i=y_train[i]
print(f"x_i={x_i}, y_i={y_i}")

plt.title('Housing Prices')
plt.scatter(x_train, y_train, marker='x', c='r')
plt.ylabel('Price in $1000s')
plt.xlabel('Size in 1000 sqft')
# plt.show()

w=200
b=100

def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb

tmp_f_wb = compute_model_output(x_train, w, b)

plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')

plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
plt.title('Housing Prices')
plt.ylabel('Price in $1000s')
plt.xlabel('Size in 1000 sqft') 
plt.legend()
plt.show()  