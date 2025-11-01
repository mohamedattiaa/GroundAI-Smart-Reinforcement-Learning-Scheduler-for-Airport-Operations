"""
Test script for RAG system components
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import unittest
import numpy as np
from phase2_rag_agent_rl.utils.data_loader import DatasetLoader, ScenarioConverter
from phase2_rag_agent_rl.rag_system.embeddings_generator import EmbeddingsGenerator
from phase2_rag_agent_rl.rag_system.vector_store import VectorStore


class TestRAGSystem(unittest.TestCase):
    """Test RAG system components"""
    
    @classmethod
    def setUpClass(cls):
        """Setup test data"""
        print("\n🧪 Setting up test environment...")
        
        # Load test scenarios
        cls.loader = DatasetLoader(data_dir="data/raw")
        cls.scenarios = cls.loader.load_scenario_files()[:5]  # Use first 5
        
        if len(cls.scenarios) == 0:
            raise ValueError("No scenarios found for testing!")
        
        print(f"✅ Loaded {len(cls.scenarios)} test scenarios")
    
    def test_embeddings_generation(self):
        """Test embedding generation"""
        print("\n🧪 Testing embeddings generation...")
        
        embeddings_gen = EmbeddingsGenerator(model_name='all-MiniLM-L6-v2')
        
        texts, embeddings, metadata = embeddings_gen.embed_scenarios(
            self.scenarios,
            batch_size=2
        )
        
        # Check outputs
        self.assertEqual(len(texts), len(self.scenarios))
        self.assertEqual(len(embeddings), len(self.scenarios))
        self.assertEqual(len(metadata), len(self.scenarios))
        
        # Check embedding shape
        self.assertEqual(embeddings.shape[0], len(self.scenarios))
        self.assertGreater(embeddings.shape[1], 0)
        
        print(f"✅ Generated {len(embeddings)} embeddings")
        print(f"   Shape: {embeddings.shape}")
    
    def test_vector_store(self):
        """Test vector store operations"""
        print("\n🧪 Testing vector store...")
        
        # Create test vector store
        vector_store = VectorStore(
            persist_directory="data/test/chroma_db",
            collection_name="test_scenarios"
        )
        
        vector_store.create_collection(reset=True)
        
        # Generate test embeddings
        embeddings_gen = EmbeddingsGenerator()
        texts, embeddings, metadata = embeddings_gen.embed_scenarios(self.scenarios)
        
        # Add to vector store
        vector_store.add_scenarios(texts, embeddings, metadata)
        
        # Check collection
        stats = vector_store.get_collection_stats()
        self.assertEqual(stats['count'], len(self.scenarios))
        
        print(f"✅ Vector store contains {stats['count']} scenarios")
    
    def test_retrieval(self):
        """Test scenario retrieval"""
        print("\n🧪 Testing retrieval...")
        
        from phase2_rag_agent_rl.rag_system.retriever import ScenarioRetriever
        
        # Setup
        vector_store = VectorStore(
            persist_directory="data/test/chroma_db",
            collection_name="test_scenarios"
        )
        vector_store.create_collection(reset=False)
        
        retriever = ScenarioRetriever(vector_store)
        
        # Test retrieval
        query = "airport scenario with many flights"
        results = retriever.retrieve(query, n_results=3)
        
        self.assertIn('query', results)
        self.assertIn('results', results)
        self.assertGreater(len(results['results']), 0)
        self.assertLessEqual(len(results['results']), 3)
        
        print(f"✅ Retrieved {len(results['results'])} scenarios")
        print(f"   Query: {query}")
    
    def test_scenario_converter(self):
        """Test scenario conversion"""
        print("\n🧪 Testing scenario converter...")
        
        scenario = self.scenarios[0]
        
        # Test text description
        text = ScenarioConverter.to_text_description(scenario)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)
        
        # Test RL state
        rl_state = ScenarioConverter.to_rl_state(scenario)
        self.assertIn('num_flights', rl_state)
        self.assertIn('num_tasks', rl_state)
        
        print(f"✅ Scenario conversion working")


def run_tests():
    """Run all tests"""
    print("="*70)
    print("RAG SYSTEM TESTS")
    print("="*70)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRAGSystem)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*70)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)