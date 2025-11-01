#!/usr/bin/env python3
"""
Build RAG system from scratch

This script:
1. Loads all scenarios
2. Generates embeddings
3. Creates vector store
4. Saves everything for later use
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from phase2_rag_agent_rl.utils.data_loader import DatasetLoader
from phase2_rag_agent_rl.rag_system.embeddings_generator import EmbeddingsGenerator
from phase2_rag_agent_rl.rag_system.vector_store import VectorStore


def build_rag_system(
    data_dir: str = "data/raw",
    scenarios_dir: str = "data/processed/scenarios",
    output_dir: str = "data/processed",
    embedding_model: str = "all-MiniLM-L6-v2"
):
    """
    Build complete RAG system
    
    Args:
        data_dir: Raw data directory
        scenarios_dir: Scenarios directory
        output_dir: Output directory for embeddings and vector store
        embedding_model: Sentence transformer model name
    """
    
    print("="*70)
    print("BUILDING RAG SYSTEM")
    print("="*70)
    
    # Step 1: Load scenarios
    print("\n📂 Step 1: Loading scenarios...")
    loader = DatasetLoader(data_dir=data_dir)
    scenarios = loader.load_scenario_files(scenarios_dir=scenarios_dir)
    
    if len(scenarios) == 0:
        print("❌ No scenarios found!")
        print(f"   Please ensure scenarios exist in: {scenarios_dir}")
        return False
    
    print(f"✅ Loaded {len(scenarios)} scenarios")
    
    # Step 2: Generate embeddings
    print(f"\n🔄 Step 2: Generating embeddings with {embedding_model}...")
    embeddings_gen = EmbeddingsGenerator(model_name=embedding_model)
    
    texts, embeddings, metadata = embeddings_gen.embed_scenarios(
        scenarios,
        batch_size=32
    )
    
    # Save embeddings
    embeddings_path = Path(output_dir) / "embeddings" / "scenario_embeddings.pkl"
    embeddings_gen.save_embeddings(texts, embeddings, metadata, str(embeddings_path))
    
    # Step 3: Build vector store
    print("\n🗄️ Step 3: Building vector store...")
    vector_store = VectorStore(
        persist_directory=str(Path(output_dir) / "chroma_db"),
        collection_name="airport_scenarios"
    )
    
    vector_store.create_collection(reset=True)
    vector_store.add_scenarios(texts, embeddings, metadata)
    
    # Step 4: Verify
    print("\n✅ Step 4: Verification...")
    stats = vector_store.get_collection_stats()
    
    print(f"   Collection: {stats['name']}")
    print(f"   Total scenarios: {stats['count']}")
    print(f"   Persist directory: {stats['persist_directory']}")
    
    # Test retrieval
    print("\n🧪 Step 5: Testing retrieval...")
    from phase2_rag_agent_rl.rag_system.retriever import ScenarioRetriever
    
    retriever = ScenarioRetriever(vector_store, model_name=embedding_model)
    
    test_query = "Scenario with 50 flights and high delays"
    results = retriever.retrieve(test_query, n_results=3)
    
    print(f"   Test query: {test_query}")
    print(f"   Retrieved: {len(results['results'])} scenarios")
    
    if results['results']:
        print(f"   Top result similarity: {1 - results['results'][0]['distance']:.3f}")
    
    print("\n" + "="*70)
    print("✅ RAG SYSTEM BUILD COMPLETE")
    print("="*70)
    print("\nYou can now use the RAG system:")
    print("  python demo_rag.py --mode interactive")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Build RAG system")
    
    parser.add_argument(
        '--data-dir',
        default='data/raw',
        help='Raw data directory'
    )
    
    parser.add_argument(
        '--scenarios-dir',
        default='data/processed/scenarios',
        help='Scenarios directory'
    )
    
    parser.add_argument(
        '--output-dir',
        default='data/processed',
        help='Output directory'
    )
    
    parser.add_argument(
        '--embedding-model',
        default='all-MiniLM-L6-v2',
        help='Sentence transformer model'
    )
    
    args = parser.parse_args()
    
    success = build_rag_system(
        data_dir=args.data_dir,
        scenarios_dir=args.scenarios_dir,
        output_dir=args.output_dir,
        embedding_model=args.embedding_model
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()