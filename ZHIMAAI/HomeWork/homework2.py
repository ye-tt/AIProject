import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def normalize_image(image):
    H, W, C = image.shape
    image_flat = image.reshape(-1, C)
    scaler  = StandardScaler()
    image_flat_norm = scaler.fit_transform(image_flat)
    image_norm = image_flat_norm.reshape(H, W, C)
    return image_norm

def cut_image(image, new_width, new_height):
    width,height = image.shape[0],image.shape[1]        
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    x1,y1,x2,y2 = int(left),int(top),int(right),int(bottom)
    cuted_image = image[x1:x2,y1:y2,:]
    return cuted_image

  
# 获取当前脚本所在目录
SCRIPT_DIR = Path(__file__).parent
image_path=SCRIPT_DIR / 'pictures' / 'dinosaur.jpg'
# 读取图像
my_image=plt.imread(image_path)
# print(my_image,my_image.shape,type(my_image))

# 对图像进行归一化处理
image_norm = normalize_image(my_image)

# 图像翻转
flipped_img = np.fliplr(my_image)

# #添加一个随机噪声（使用 np.random.normal，标准差为 0.1）
noise = np.random.normal(0, 0.1, my_image.shape)
noisy_img = np.clip(my_image + noise, 0, 1)  # 裁剪到 [0,1]

# ##将图像裁剪为 24×24（居中裁剪）
cuted_image = cut_image(my_image,24,24)
# print(cuted_image,cuted_image.shape,type(cuted_image))

result_img=cuted_image[:,:,0:1]
result_array=result_img.reshape(1,24,24)
print(result_array,result_array.shape,type(result_array))

fig, ax = plt.subplots(1, 4, figsize=(12, 3))
ax[0].set_title("Original Image")
ax[0].imshow(my_image)
ax[1].set_title("Flipped Image")
ax[1].imshow(flipped_img)
ax[2].set_title("Gaussian Noise Image")
ax[2].imshow(noisy_img)
ax[3].set_title("Cuted Image")
ax[3].imshow(cuted_image)
plt.show()



