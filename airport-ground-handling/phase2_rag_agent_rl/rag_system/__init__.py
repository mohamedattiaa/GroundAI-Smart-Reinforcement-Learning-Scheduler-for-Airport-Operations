"""
RAG System for Airport Ground Handling
Retrieval-Augmented Generation system for scenario retrieval and knowledge augmentation
"""

__version__ = "1.0.0"

# Import all RAG components
try:
    from .vector_store import VectorStore
except ImportError as e:
    print(f"Warning: Could not import VectorStore: {e}")
    VectorStore = None

try:
    # Import ScenarioRetriever and alias it as Retriever for compatibility
    from .retriever import ScenarioRetriever
    Retriever = ScenarioRetriever  # Alias for backward compatibility
except ImportError as e:
    print(f"Warning: Could not import Retriever: {e}")
    Retriever = None
    ScenarioRetriever = None

try:
    from .embeddings_generator import EmbeddingsGenerator
    EmbeddingsEngine = EmbeddingsGenerator  # Alias for backward compatibility
except ImportError as e:
    print(f"Warning: Could not import EmbeddingsGenerator: {e}")
    EmbeddingsGenerator = None
    EmbeddingsEngine = None

try:
    from .rag_query_engine import RAGQueryEngine
except ImportError as e:
    print(f"Warning: Could not import RAGQueryEngine: {e}")
    RAGQueryEngine = None

try:
    from .rag_query_engine_streaming import RAGQueryEngineStreaming
except ImportError as e:
    print(f"Warning: Could not import RAGQueryEngineStreaming: {e}")
    RAGQueryEngineStreaming = None

# Export public API
__all__ = [
    'VectorStore',
    'Retriever',
    'ScenarioRetriever',
    'EmbeddingsEngine',
    'EmbeddingsGenerator',
    'RAGQueryEngine',
    'RAGQueryEngineStreaming',
]