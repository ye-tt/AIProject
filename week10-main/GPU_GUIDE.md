# GPU 运行配置指南

## 修改说明

已修改代码以支持GPU运行。主要改动如下：

### 1. **embeddings.py 更新**

- ✅ 添加了 `torch` 导入以检测GPU
- ✅ `EmbeddingService.__init__()` 新增 `device` 参数，自动检测GPU
  - 如果有CUDA设备可用，默认使用 `cuda`
  - 否则降级为 `cpu`
  
- ✅ `embed_documents()` 和 `embed_query()` 新增 `device` 参数传递给模型
- ✅ 新增 `get_device_info()` 方法，可查询当前设备信息

### 2. **requirements.txt 更新**

添加了 PyTorch 依赖：
```
torch>=2.0.0
```

## 使用方法

### 方法1：自动检测GPU（推荐）

```python
from app.services.embeddings import get_embedding_service

# 自动检测GPU，如果可用则使用GPU，否则使用CPU
embedding_service = get_embedding_service()

# 查看当前运行设备
device_info = embedding_service.get_device_info()
print(device_info)
# 输出示例：
# {
#     "device": "cuda",
#     "cuda_available": True,
#     "cuda_device_count": 1,
#     "current_device": 0
# }
```

### 方法2：强制指定设备

```python
from app.services.embeddings import EmbeddingService

# 强制使用GPU
embedding_service = EmbeddingService(device="cuda")

# 或强制使用CPU
embedding_service = EmbeddingService(device="cpu")
```

### 方法3：指定GPU设备

```python
from app.services.embeddings import EmbeddingService

# 使用特定GPU设备（如果有多张GPU）
embedding_service = EmbeddingService(device="cuda:0")  # 第一张GPU
embedding_service = EmbeddingService(device="cuda:1")  # 第二张GPU
```

## 安装步骤

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 如果没有安装CUDA相关驱动

#### Windows用户：

选择对应的PyTorch版本安装：

```bash
# CUDA 12.1 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Linux用户：

```bash
# CUDA 12.1 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. 验证GPU设置

运行测试脚本：

```bash
python gpu_config.py
```

输出示例：
```
CUDA Available: True
CUDA Device Count: 1
Current CUDA Device: 0
CUDA Device Name: NVIDIA GeForce RTX 4090
CUDA Capability: (8, 9)

--- Embedding Service Device Info ---
device: cuda
cuda_available: True
cuda_device_count: 1
current_device: 0

--- Testing Embeddings on cuda ---
Embeddings shape: (3, 1024)
Embeddings dtype: float32
```

## 性能建议

### 批处理优化

处理大量文本时，建议使用批处理以充分利用GPU：

```python
embedding_service = get_embedding_service()

# 批量处理文档
batch_size = 128
texts = [...]  # 大量文本列表

for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    embeddings = embedding_service.embed_documents(batch)
    # 处理embeddings...
```

### 内存管理

如果遇到OOM错误，可以：

1. 减小批处理大小
2. 使用 `torch.cuda.empty_cache()` 清理显存
3. 指定使用CPU处理

```python
import torch

# 清理显存
torch.cuda.empty_cache()

# 如果还是OOM，切换到CPU
embedding_service = EmbeddingService(device="cpu")
```

## 故障排查

### 问题1：GPU未被检测到

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

如果输出 `False`，需要：
1. 检查NVIDIA驱动是否正确安装
2. 重新安装对应CUDA版本的PyTorch

### 问题2：CUDA内存错误

减小批处理大小或切换到CPU模式

### 问题3：导入错误

确保已安装所有依赖：
```bash
pip install -r requirements.txt
```
