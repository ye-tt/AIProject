import faiss
from pathlib import Path

path = Path("storage/faiss.index")

print(f"文件路径: {path}")
print(f"文件存在: {path.exists()}")
print(f"文件大小: {path.stat().st_size} 字节")
print(f"文件可读: {path.is_file()}")

# try:
#     index = faiss.read_index(str(path))
#     print("✅ 成功加载索引！")
# except Exception as e:
#     print(f"❌ 加载失败: {e}")