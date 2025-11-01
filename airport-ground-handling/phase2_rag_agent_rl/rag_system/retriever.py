"""
Retriever for RAG system
"""

from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import numpy as np
from .vector_store import VectorStore


class ScenarioRetriever:
    """Retrieve similar scenarios for RAG"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        model_name: str = 'all-MiniLM-L6-v2'
    ):
        """
        Initialize retriever
        
        Args:
            vector_store: VectorStore instance
            model_name: Embedding model name
        """
        self.vector_store = vector_store
        self.model = SentenceTransformer(model_name)
        
        print(f"✅ Retriever initialized with model: {model_name}")
    
    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        filters: Optional[Dict] = None
    ) -> Dict:
        """
        Retrieve similar scenarios
        
        Args:
            query: Natural language query
            n_results: Number of results to return
            filters: Optional metadata filters
        
        Returns:
            Dictionary with query and results
        """
        # Generate query embedding
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        # Search vector store
        results = self.vector_store.search(
            query_text=query,
            query_embedding=query_embedding,
            n_results=n_results,
            where_filter=filters
        )
        
        return results
    
    def retrieve_by_criteria(
        self,
        num_flights: Optional[int] = None,
        aircraft_types: Optional[List[str]] = None,
        min_delay: Optional[float] = None,
        max_delay: Optional[float] = None,
        n_results: int = 5
    ) -> List[Dict]:
        """
        Retrieve scenarios matching specific criteria
        
        Args:
            num_flights: Target number of flights
            aircraft_types: List of aircraft types to include
            min_delay: Minimum average delay
            max_delay: Maximum average delay
            n_results: Number of results
        
        Returns:
            List of matching scenarios
        """
        # Build query text
        query_parts = []
        
        if num_flights:
            query_parts.append(f"{num_flights} flights")
        
        if aircraft_types:
            query_parts.append(f"aircraft types {', '.join(aircraft_types)}")
        
        if min_delay or max_delay:
            if min_delay and max_delay:
                query_parts.append(f"delay between {min_delay} and {max_delay} minutes")
            elif min_delay:
                query_parts.append(f"delay at least {min_delay} minutes")
            else:
                query_parts.append(f"delay at most {max_delay} minutes")
        
        query = "Airport scenario with " + " ".join(query_parts)
        
        # Retrieve
        results = self.retrieve(query, n_results=n_results)
        
        return results['results']
    
    def format_results(self, results: Dict) -> str:
        """Format retrieval results as readable text"""
        
        output = [f"Query: {results['query']}\n"]
        output.append("=" * 60)
        
        for i, result in enumerate(results['results'], 1):
            output.append(f"\nResult {i}:")
            output.append(f"  Similarity: {1 - result['distance']:.3f}")
            output.append(f"  {result['document']}")
            
            if 'metadata' in result:
                meta = result['metadata']
                # Access flattened statistics
                total_tasks = meta.get('total_tasks', 'N/A')
                avg_delay = meta.get('avg_delay', 0)
                equipment_failures = meta.get('equipment_failures', 0)
                
                output.append(
                    f"  Stats: {total_tasks} tasks, "
                    f"{avg_delay:.1f}min avg delay, "
                    f"{equipment_failures} failures"
                )
        
        return "\n".join(output)