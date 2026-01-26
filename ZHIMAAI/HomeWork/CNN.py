import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def apply_conv_and_show(image_path):
    # === 步骤1: 读取图像并转灰度 ===
    img_np = plt.imread(image_path)
    if len(img_np.shape) == 3:
        img_gray = np.dot(img_np[...,:3], [0.2989, 0.5870, 0.1140])  # RGB to Gray
    else:
        img_gray = img_np
    img_tensor = torch.tensor(img_gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    #torch.clamp() 功能与 NumPy 中的 np.clip() 非常相似，常用于梯度裁剪、激活值限制、图像像素归一化后处理等场景
    # torch.clamp(x, min, max)	限制值在 [min, max]
    img_tensor = torch.clamp(img_tensor, 0, 1)

    print("Original image shape:", img_tensor.shape)

    # === 步骤2: 定义卷积核 ===
    # 示例：随机核（用于演示），也可自定义常用卷积核
    #torch.rand 是 PyTorch 中用于生成服从均匀分布的随机张量的核心函数，常用于初始化权重、创建随机输入、测试模型等场景。
    # kernel_3x3 = torch.rand(1, 1, 3, 3, requires_grad=False)
    kernel_3x3 = torch.tensor([[[[0, -1, 0], [-1, 5, -1], [0, -1, 0]]]], dtype=torch.float32)
    kernel_5x5 = torch.rand(1, 1, 5, 5, requires_grad=False)

    # print(kernel_3x3,kernel_3x3.shape,type(kernel_3x3))

    # === 步骤3: 执行卷积 (保持输出尺寸一致) ===
    conv3_out = F.conv2d(img_tensor, kernel_3x3, padding=1, stride=1)
    conv5_out = F.conv2d(img_tensor, kernel_5x5, padding=2, stride=1)

    print("3x3 Conv output shape:", conv3_out.shape)
    print("5x5 Conv output shape:", conv5_out.shape)

    # === 步骤4: 归一化结果便于显示 ===
    def normalize(tensor):
        arr = tensor[0, 0].detach().numpy()
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return arr


    conv3_vis = normalize(conv3_out)
    conv5_vis = normalize(conv5_out)
    
    print("3x3 Conv output shape:", conv3_vis.shape)
    print("5x5 Conv output shape:", conv5_vis.shape)
    # === 步骤5: 显示结果 ===
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img_gray, cmap='gray')
    axs[0].set_title('Original Local Image')
    axs[0].axis('off')

    axs[1].imshow(conv3_out[0,0], cmap='gray')
    axs[1].set_title('3x3 Convolution')
    axs[1].axis('off')

    axs[2].imshow(conv5_vis, cmap='gray')
    axs[2].set_title('5x5 Convolution')
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

# === 调用示例 ===
SCRIPT_DIR = Path(__file__).parent
image_path=SCRIPT_DIR / 'pictures' / 'dinosaur.jpg'
apply_conv_and_show(image_path)  # 替换为你的本地图像路径
