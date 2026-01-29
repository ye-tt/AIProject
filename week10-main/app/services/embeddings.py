from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Union

import torch
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
from app.core.config import settings


MODEL_ALIASES = {
    "all-minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "all-minilm-l6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-minilm-l6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "clip": "sentence-transformers/clip-ViT-B-32",
    "clip-vit-b-32": "sentence-transformers/clip-ViT-B-32",
    "sentence-transformers/clip-vit-b-32": "sentence-transformers/clip-ViT-B-32",
    "openai/clip-vit-base-patch32": "openai/clip-vit-base-patch32",
}

class EmbeddingService:
    """Encapsulates the embedding model lifecycle."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        # Auto-detect GPU if not specified
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = self._resolve_model_name(model_name)
        self._st_model: SentenceTransformer | None = None
        self._clip_model: CLIPModel | None = None
        self._clip_processor: CLIPProcessor | None = None
        self._is_local_clip = "openai/clip-vit-base-patch32" in self.model_name or "clip-vit-base-patch32" in self.model_name

    @staticmethod
    def _resolve_model_name(name: str | None) -> str:
        base = name or settings.model_name
        key = base.lower()
        if key in MODEL_ALIASES:
            return MODEL_ALIASES[key]
        return base

    def _get_local_clip_path(self):
        # Assuming the model is in models/clip-vit-base-patch32 relative to project root
        base_dir = Path(__file__).resolve().parents[2]
        model_path = base_dir / "models" / "clip-vit-base-patch32"
        if model_path.exists():
            return str(model_path)
        return self.model_name
    #
    @property
    def model(self):
        if self._is_local_clip:
            if self._clip_model is None:
                path = self._get_local_clip_path()
                print(f"Loading local CLIP model from: {path}")
                self._clip_model = CLIPModel.from_pretrained(path)
                self._clip_processor = CLIPProcessor.from_pretrained(path)
            return self._clip_model
        else:
            if self._st_model is None:
                self._st_model = SentenceTransformer(self.model_name)
            return self._st_model

    def embed_documents(self, texts: Iterable[str]) -> np.ndarray:
        if self._is_local_clip:
            # Ensure model and processor are loaded
            _ = self.model

            inputs = self._clip_processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
            # Normalize embeddings
            embeddings = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            return embeddings.numpy().astype("float32")
        else:
            embeddings = self.model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
            return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_documents([query])

    def embed_image(self, image_source: Union[str, Path, Image.Image]) -> np.ndarray:
        """Generate embedding for an image."""
        if isinstance(image_source, (str, Path)):
            image = Image.open(image_source)
        else:
            image = image_source

        if self._is_local_clip:
            # Ensure model and processor are loaded
            _ = self.model

            inputs = self._clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            # Normalize embeddings 对嵌入向量（embeddings）进行 L2 归一化（L2 normalization）
            '''
            假设 outputs 是一个形状为 (batch_size, embedding_dim) 的张量
            计算每个向量（即每行）的 L2 范数（欧几里得范数）
            p=2 表示 L2 范数：x₂ = √(x₁² + x₂² + ... + xₙ²)
            dim=-1 表示在最后一个维度（即 embedding 维度）上计算范数。
            keepdim=True 保留维度，使结果形状为 (batch_size, 1)，便于广播除法。
            outputs / ...：
            将每个向量除以其 L2 范数，得到 单位向量（unit vector），即模长为 1 的向量。
            '''
            embeddings = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            return embeddings.numpy().astype("float32")
        else:
            embeddings = self.model.encode(image, convert_to_numpy=True, normalize_embeddings=True)
            return embeddings.reshape(1, -1).astype("float32")
    
    def get_device_info(self) -> dict:
        """Get current device information."""
        return {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None
        }


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


if __name__ == "__main__":
    # Test case for local CLIP model
    print("-" * 50)
    print("Testing local CLIP model...")
    service = EmbeddingService("openai/clip-vit-base-patch32")

    # Text embedding
    text = "The image depicts a formal military parade with a group of uniformed personnel marching in formation. They are dressed in dark blue uniforms with white gloves and black boots, carrying rifles. The soldiers are marching in unison, showcasing discipline and precision. In the background, several flags are visible, including the Malaysian flag, indicating the event is likely taking place in Malaysia. The setting appears to be an urban area with modern buildings and trees lining the street. The atmosphere suggests a significant national or ceremonial occasion."
    text_emb = service.embed_query(text)
    print(f"Text embedding shape: {text_emb.shape}")
    print(f"Text embedding snippet: {text_emb[0][:5]}")

    # Image embedding
    # Find an image from the small dataset
    data_dir = Path(__file__).resolve().parents[2] / "data" / "small_dataset" / "images"
    image_files = list(data_dir.glob("*0008.png"))

    if image_files:
        image_path = image_files[0]
        print(f"Testing with image: {image_path}")
        image_emb = service.embed_image(image_path)
        print(f"Image embedding shape: {image_emb.shape}")
        print(f"Image embedding snippet: {image_emb[0][:5]}")

        # Verify normalization (should be close to 1)
        norm = np.linalg.norm(image_emb)
        print(f"Image embedding norm: {norm}")
    else:
        print("No images found in small_dataset to test.")