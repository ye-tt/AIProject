from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import numpy as np
from sklearn.pipeline import make_pipeline

from app.models.document import Document
from app.services.embeddings import EmbeddingService
from app.services.knowledge_base import KnowledgeBase
from app.services.parser import parse_directory


class DocumentParser():
    """Parse text files into document chunks."""
    
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, source_dirs: List[Path]) -> List[Document]:
        """Parse directories and return document chunks."""
        documents = []
        for source_dir in source_dirs:
            docs = parse_directory(
                directory=source_dir,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            documents.extend(docs)
        return documents


class EmbeddingTransformer():
    """Generate embeddings for documents."""
    
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, documents: List[Document]) -> tuple[np.ndarray, List[Document]]:
        """Generate embeddings for documents."""
        texts = [doc.text for doc in documents]
        embeddings = self.embedding_service.embed_documents(texts)
        embeddings = np.array(embeddings, dtype=np.float32)
        return embeddings, documents


class KnowledgeBaseIndexer():
    """Index embeddings and metadata into knowledge base."""
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, data: tuple[np.ndarray, List[Document]]) -> dict:
        """Store embeddings and documents in knowledge base."""
        embeddings, documents = data
        ids = self.knowledge_base.add_documents(embeddings, documents)
        
        return {
            "status": "success",
            "documents_count": len(documents),
            "embedding_dimension": embeddings.shape[1],
            "document_ids": ids,
            "index_path": str(self.knowledge_base.index_path),
            "metadata_path": str(self.knowledge_base.metadata_path),
        }


class DataProcessingPipeline:
    """Data processing pipeline using sklearn make_pipeline."""
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        knowledge_base: KnowledgeBase,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
    ):
        self.embedding_service = embedding_service
        self.knowledge_base = knowledge_base
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.pipeline = self._build_pipeline()
    
    def _build_pipeline(self):
        """Build sklearn pipeline using make_pipeline."""
        pipeline = make_pipeline(
            DocumentParser(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            ),
            EmbeddingTransformer(
                embedding_service=self.embedding_service,
            ),
            KnowledgeBaseIndexer(
                knowledge_base=self.knowledge_base,
            ),
            verbose=True,
        )
        return pipeline
    
    def process(self, source_dirs: List[Path] | Path) -> dict:
        """Execute the complete data processing pipeline."""
        if isinstance(source_dirs, Path):
            source_dirs = [source_dirs]
        
        print(f"Starting pipeline with {len(source_dirs)} source directories")
        
        # Execute pipeline
        result = self.pipeline.fit_transform(source_dirs)
        
        print(f"Pipeline completed: {result}")
        return result
    
    def get_pipeline_steps(self) -> List[tuple]:
        """Get pipeline steps for inspection."""
        return self.pipeline.steps