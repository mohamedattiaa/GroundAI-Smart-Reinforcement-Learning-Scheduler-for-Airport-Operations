"""
Vector store for RAG system using ChromaDB
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import numpy as np
from pathlib import Path


class VectorStore:
    """ChromaDB vector store for scenario retrieval"""
    
    def __init__(
        self,
        persist_directory: str = "data/processed/chroma_db",
        collection_name: str = "airport_scenarios"
    ):
        """
        Initialize vector store
        
        Args:
            persist_directory: Directory to persist ChromaDB
            collection_name: Name of the collection
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )
        
        self.collection_name = collection_name
        self.collection = None
        
        print(f"✅ Vector store initialized at {persist_directory}")
    
    def create_collection(self, reset: bool = False):
        """Create or get collection"""
        
        if reset and self.collection_name in [c.name for c in self.client.list_collections()]:
            self.client.delete_collection(self.collection_name)
            print(f"🗑️  Deleted existing collection: {self.collection_name}")
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"✅ Collection ready: {self.collection_name}")
        print(f"   Current size: {self.collection.count()}")
    
    def add_scenarios(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadata: List[Dict],
        ids: Optional[List[str]] = None
    ):
        """
        Add scenarios to vector store
        
        Args:
            texts: List of scenario text descriptions
            embeddings: Numpy array of embeddings
            metadata: List of metadata dicts
            ids: Optional list of IDs (will generate if None)
        """
        if self.collection is None:
            self.create_collection()
        
        if ids is None:
            ids = [f"scenario_{i:04d}" for i in range(len(texts))]
        
        # Convert numpy to list for ChromaDB
        embeddings_list = embeddings.tolist()
        
        print(f"\n📥 Adding {len(texts)} scenarios to vector store...")
        
        # Add in batches
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_end = min(i + batch_size, len(texts))
            
            self.collection.add(
                documents=texts[i:batch_end],
                embeddings=embeddings_list[i:batch_end],
                metadatas=metadata[i:batch_end],
                ids=ids[i:batch_end]
            )
        
        print(f"✅ Added {len(texts)} scenarios")
        print(f"   Total in collection: {self.collection.count()}")
    
    def search(
        self,
        query_text: str,
        query_embedding: Optional[np.ndarray] = None,
        n_results: int = 5,
        where_filter: Optional[Dict] = None
    ) -> Dict:
        """
        Search for similar scenarios
        
        Args:
            query_text: Query text (for display)
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where_filter: Optional metadata filter
        
        Returns:
            Search results dictionary
        """
        if self.collection is None:
            raise ValueError("Collection not initialized. Call create_collection() first.")
        
        if query_embedding is None:
            raise ValueError("query_embedding is required")
        
        # Convert to list
        query_embedding_list = query_embedding.tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding_list],
            n_results=n_results,
            where=where_filter,
            include=['documents', 'metadatas', 'distances']
        )
        
        return {
            'query': query_text,
            'results': [
                {
                    'document': doc,
                    'metadata': meta,
                    'distance': dist
                }
                for doc, meta, dist in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )
            ]
        }
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        if self.collection is None:
            return {'error': 'Collection not initialized'}
        
        return {
            'name': self.collection_name,
            'count': self.collection.count(),
            'persist_directory': str(self.persist_directory)
        }