#!/usr/bin/env python3
"""
Demo script for RAG system

This demonstrates:
1. Loading dataset
2. Generating embeddings
3. Building vector store
4. Querying with natural language
5. Getting recommendations
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from phase2_rag_agent_rl.utils.data_loader import DatasetLoader, ScenarioConverter
from phase2_rag_agent_rl.rag_system.embeddings_generator import EmbeddingsGenerator
from phase2_rag_agent_rl.rag_system.vector_store import VectorStore
from phase2_rag_agent_rl.rag_system.retriever import ScenarioRetriever
from phase2_rag_agent_rl.rag_system.rag_query_engine import RAGQueryEngine


def setup_rag_system(force_rebuild: bool = False):
    """
    Setup complete RAG system
    
    Args:
        force_rebuild: If True, rebuild embeddings and vector store
    
    Returns:
        RAGQueryEngine instance
    """
    print("="*70)
    print("SETTING UP RAG SYSTEM")
    print("="*70)
    
    # Step 1: Load data
    print("\n📂 Step 1: Loading dataset...")
    loader = DatasetLoader(data_dir="data/raw")
    scenarios = loader.load_scenario_files()
    
    print(f"   Loaded {len(scenarios)} scenarios")
    
    # Step 2: Generate embeddings (or load cached)
    embeddings_path = Path("data/processed/embeddings/scenario_embeddings.pkl")
    
    if force_rebuild or not embeddings_path.exists():
        print("\n🔄 Step 2: Generating embeddings...")
        embeddings_gen = EmbeddingsGenerator(model_name='all-MiniLM-L6-v2')
        
        texts, embeddings, metadata = embeddings_gen.embed_scenarios(scenarios)
        
        # Save embeddings
        embeddings_gen.save_embeddings(texts, embeddings, metadata)
    else:
        print("\n📥 Step 2: Loading cached embeddings...")
        embeddings_data = EmbeddingsGenerator.load_embeddings()
        texts = embeddings_data['texts']
        embeddings = embeddings_data['embeddings']
        metadata = embeddings_data['metadata']
    
    # Step 3: Create vector store
    print("\n🗄️ Step 3: Setting up vector store...")
    vector_store = VectorStore(
        persist_directory="data/processed/chroma_db",
        collection_name="airport_scenarios"
    )
    
    vector_store.create_collection(reset=force_rebuild)
    
    if force_rebuild or vector_store.collection.count() == 0:
        print("   Adding scenarios to vector store...")
        vector_store.add_scenarios(texts, embeddings, metadata)
    else:
        print(f"   Using existing collection with {vector_store.collection.count()} scenarios")
    
    # Step 4: Create retriever
    print("\n🔍 Step 4: Creating retriever...")
    retriever = ScenarioRetriever(
        vector_store=vector_store,
        model_name='all-MiniLM-L6-v2'
    )
    
    # Step 5: Create RAG query engine
    print("\n🤖 Step 5: Initializing RAG Query Engine...")
    rag_engine = RAGQueryEngine(
        retriever=retriever,
        llm_url="http://localhost:11434/api/generate",
        model_name="mistral:7b-instruct"
    )
    
    print("\n✅ RAG system ready!")
    print("="*70)
    
    return rag_engine


def demo_basic_queries(rag_engine: RAGQueryEngine):
    """Demo basic question answering"""
    
    print("\n" + "="*70)
    print("DEMO: Basic Queries")
    print("="*70)
    
    queries = [
        "How many vehicles do I need for 50 flights in 2 hours?",
        "What causes the most delays in wide-body aircraft operations?",
        "How to optimize turnaround for narrow-body aircraft?",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'─'*70}")
        print(f"Query {i}: {query}")
        print('─'*70)
        
        # Use shorter timeout for demo
        result = rag_engine.query(query, n_retrieval=2)  # Only 2 scenarios for speed
        
        print("\n💡 Answer:")
        print(result['answer'])
        
        print(f"\n📚 Based on {result['num_retrieved']} similar scenarios")
        
        # Only wait for user on last query
        if i < len(queries):
            print("\n⏭️  Moving to next query in 3 seconds...")
            import time
            time.sleep(3)
        else:
            input("\n\nPress Enter to finish...")

def demo_scenario_analysis(rag_engine: RAGQueryEngine):
    """Demo scenario analysis"""
    
    print("\n" + "="*70)
    print("DEMO: Scenario Analysis")
    print("="*70)
    
    # Load a sample scenario
    loader = DatasetLoader(data_dir="data/raw")
    scenarios = loader.load_scenario_files()
    
    sample_scenario = scenarios[0]
    
    print(f"\n📋 Analyzing scenario: {sample_scenario['scenario_id']}")
    print(f"   Flights: {sample_scenario['num_flights']}")
    print(f"   Tasks: {sample_scenario['statistics']['total_tasks']}")
    print(f"   Avg delay: {sample_scenario['statistics']['avg_delay']:.2f} min")
    
    # Analyze
    analysis = rag_engine.analyze_scenario(sample_scenario)
    
    print("\n📊 Analysis:")
    print(analysis['analysis'])
    
    print(f"\n🔗 Compared with {len(analysis['similar_scenarios'])} similar scenarios")


def demo_vehicle_recommendation(rag_engine: RAGQueryEngine):
    """Demo vehicle allocation recommendation"""
    
    print("\n" + "="*70)
    print("DEMO: Vehicle Allocation Recommendation")
    print("="*70)
    
    # Example scenario
    num_flights = 30
    aircraft_types = ['A320', 'B737', 'B777']
    time_window = 120  # minutes
    
    print(f"\n📝 Scenario:")
    print(f"   Flights: {num_flights}")
    print(f"   Aircraft: {', '.join(aircraft_types)}")
    print(f"   Time window: {time_window} minutes")
    
    # Get recommendation
    recommendation = rag_engine.recommend_vehicle_allocation(
        num_flights=num_flights,
        aircraft_types=aircraft_types,
        time_window_minutes=time_window
    )
    
    print("\n🎯 Recommendation:")
    print(recommendation['recommendation'])


def demo_interactive_mode(rag_engine: RAGQueryEngine):
    """Demo interactive Q&A"""
    
    rag_engine.interactive_mode()


def main():
    """Main demo function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG System Demo")
    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Force rebuild embeddings and vector store'
    )
    parser.add_argument(
        '--mode',
        choices=['basic', 'analysis', 'vehicle', 'interactive', 'all'],
        default='all',
        help='Demo mode to run'
    )
    
    args = parser.parse_args()
    
    # Setup RAG system
    rag_engine = setup_rag_system(force_rebuild=args.rebuild)
    
    # Run demos based on mode
    if args.mode in ['basic', 'all']:
        demo_basic_queries(rag_engine)
    
    if args.mode in ['analysis', 'all']:
        demo_scenario_analysis(rag_engine)
    
    if args.mode in ['vehicle', 'all']:
        demo_vehicle_recommendation(rag_engine)
    
    if args.mode == 'interactive':
        demo_interactive_mode(rag_engine)
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE")
    print("="*70)
    print("\nTo run interactive mode:")
    print("  python demo_rag.py --mode interactive")
    print("\nTo rebuild embeddings:")
    print("  python demo_rag.py --rebuild")


if __name__ == "__main__":
    main()