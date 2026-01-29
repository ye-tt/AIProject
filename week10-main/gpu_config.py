"""GPU Configuration and Utilities"""
import torch
from app.services.embeddings import get_embedding_service


def check_gpu_availability():
    """检查GPU是否可用"""
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")


def test_embedding_on_gpu():
    """测试在GPU上运行Embedding模型"""
    embedding_service = get_embedding_service()
    print(f"\n--- Embedding Service Device Info ---")
    device_info = embedding_service.get_device_info()
    for key, value in device_info.items():
        print(f"{key}: {value}")
    
    # 测试文本
    test_texts = [
        "这是一个测试文本",
        "GPU加速模型运行",
        "向量化表示"
    ]
    
    print(f"\n--- Testing Embeddings on {embedding_service.device} ---")
    embeddings = embedding_service.embed_documents(test_texts)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings dtype: {embeddings.dtype}")


if __name__ == "__main__":
    check_gpu_availability()
    test_embedding_on_gpu()
