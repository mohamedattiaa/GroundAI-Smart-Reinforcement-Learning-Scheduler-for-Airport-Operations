"""
Generate embeddings from scenarios for RAG system
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from tqdm import tqdm
import pickle
from pathlib import Path


class EmbeddingsGenerator:
    """Generate embeddings for scenarios"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embeddings generator
        
        Args:
            model_name: Sentence transformer model name
        """
        print(f"📥 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print(f"✅ Model loaded, embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def generate_scenario_text(self, scenario: Dict) -> str:
        """Convert scenario to searchable text"""
        
        parts = []
        
        # Basic info
        parts.append(f"Scenario {scenario.get('scenario_id', 'unknown')}")
        parts.append(f"{scenario['num_flights']} flights")
        
        # Time window
        if 'time_window' in scenario:
            parts.append(
                f"from {scenario['time_window']['start']} "
                f"to {scenario['time_window']['end']}"
            )
        
        # Aircraft mix
        if 'aircraft_mix' in scenario:
            aircraft_desc = ", ".join([
                f"{count} {ac_type}" 
                for ac_type, count in scenario['aircraft_mix'].items()
            ])
            parts.append(f"Aircraft: {aircraft_desc}")
        
        # Statistics
        if 'statistics' in scenario:
            stats = scenario['statistics']
            parts.append(f"{stats['total_tasks']} tasks")
            parts.append(f"average delay {stats['avg_delay']:.1f} minutes")
            if stats['equipment_failures'] > 0:
                parts.append(f"{stats['equipment_failures']} equipment failures")
        
        return " ".join(parts)
    
    def embed_scenarios(
        self, 
        scenarios: List[Dict],
        batch_size: int = 32
    ) -> tuple:
        """
        Generate embeddings for multiple scenarios
        
        Args:
            scenarios: List of scenario dictionaries
            batch_size: Batch size for embedding generation
        
        Returns:
            Tuple of (texts, embeddings, metadata)
        """
        print(f"\n🔄 Generating embeddings for {len(scenarios)} scenarios...")
        
        texts = []
        metadata = []
        
        for scenario in scenarios:
            text = self.generate_scenario_text(scenario)
            texts.append(text)
            
            # Flatten metadata - ChromaDB doesn't support nested dicts
            flat_metadata = {
                'scenario_id': str(scenario.get('scenario_id', 'unknown')),
                'num_flights': int(scenario['num_flights']),
            }
            
            # Flatten statistics into top-level keys
            if 'statistics' in scenario and scenario['statistics']:
                stats = scenario['statistics']
                flat_metadata['total_tasks'] = int(stats.get('total_tasks', 0))
                flat_metadata['avg_delay'] = float(stats.get('avg_delay', 0.0))
                flat_metadata['equipment_failures'] = int(stats.get('equipment_failures', 0))
            else:
                flat_metadata['total_tasks'] = 0
                flat_metadata['avg_delay'] = 0.0
                flat_metadata['equipment_failures'] = 0
            
            # Add aircraft mix as separate keys (limited to simple types)
            if 'aircraft_mix' in scenario and scenario['aircraft_mix']:
                for aircraft, count in list(scenario['aircraft_mix'].items())[:5]:  # Limit to 5 types
                    # Clean aircraft name for key
                    safe_key = f"ac_{aircraft.replace('-', '_')}"
                    flat_metadata[safe_key] = int(count)
            
            metadata.append(flat_metadata)
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"✅ Generated {len(embeddings)} embeddings")
        print(f"   Shape: {embeddings.shape}")
        
        return texts, embeddings, metadata
    
    def save_embeddings(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadata: List[Dict],
        output_path: str = "data/processed/embeddings/scenario_embeddings.pkl"
    ):
        """Save embeddings to disk"""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'texts': texts,
            'embeddings': embeddings,
            'metadata': metadata,
            'model_name': self.model_name
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ Embeddings saved to {output_path}")
    
    @staticmethod
    def load_embeddings(
        embeddings_path: str = "data/processed/embeddings/scenario_embeddings.pkl"
    ) -> Dict:
        """Load pre-generated embeddings"""
        
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ Loaded embeddings from {embeddings_path}")
        print(f"   Model: {data['model_name']}")
        print(f"   Count: {len(data['texts'])}")
        
        return data