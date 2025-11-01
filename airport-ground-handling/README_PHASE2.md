# RAG System for Airport Ground Handling

## Overview

This RAG (Retrieval-Augmented Generation) system enables intelligent querying of historical airport operations data to provide recommendations for scheduling and resource allocation.

## Features

- **Semantic Search**: Find similar historical scenarios using sentence embeddings
- **Vector Store**: ChromaDB-based persistent storage for fast retrieval
- **LLM Integration**: Ollama-powered answer generation
- **Multiple Query Modes**: Basic Q&A, scenario analysis, vehicle recommendations
- **Interactive Mode**: Chat-like interface for exploring the data

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_phase2.txt
```

### 2. Build RAG System
```bash
# Build from your generated dataset
python build_rag_system.py

# This will:
# - Load all scenario files
# - Generate embeddings (takes ~5 minutes for 100 scenarios)
# - Create vector store
# - Save for reuse
```

### 3. Run Demo
```bash
# Run all demos
python demo_rag.py

# Or specific modes:
python demo_rag.py --mode basic      # Basic Q&A
python demo_rag.py --mode analysis   # Scenario analysis
python demo_rag.py --mode vehicle    # Vehicle recommendations
python demo_rag.py --mode interactive # Interactive chat
```

## Architecture
```
User Query
    ↓
[Query Embedding]
    ↓
[Vector Store Search] → Retrieve similar scenarios
    ↓
[Prompt Construction] → Add context + question
    ↓
[LLM Generation] → Generate answer
    ↓
Structured Response
```

## Usage Examples

### Example 1: Basic Query
```python
from demo_rag import setup_rag_system

# Setup
rag_engine = setup_rag_system()

# Query
result = rag_engine.query(
    "How many fuel trucks needed for 50 flights in 2 hours?"
)

print(result['answer'])
```

### Example 2: Scenario Analysis
```python
# Analyze a specific scenario
scenario = {...}  # Your scenario data

analysis = rag_engine.analyze_scenario(scenario)
print(analysis['analysis'])
```

### Example 3: Vehicle Recommendation
```python
# Get vehicle allocation recommendation
recommendation = rag_engine.recommend_vehicle_allocation(
    num_flights=30,
    aircraft_types=['A320', 'B737'],
    time_window_minutes=120
)

print(recommendation['recommendation'])
```

## Configuration

Edit `configs/rag_config.yaml`:
```yaml
rag_system:
  embedding_model: "all-MiniLM-L6-v2"
  
  retrieval:
    default_k: 5
    min_similarity: 0.5
  
  llm:
    model_name: "mistral:7b-instruct"
    temperature: 0.7
```

## LLM Setup (Ollama)

### Install Ollama
```bash
# Linux/Mac
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.ai
```

### Pull Model
```bash
ollama pull mistral:7b-instruct
```

### Verify
```bash
ollama list
# Should show mistral:7b-instruct
```

## Performance

- **Embedding generation**: ~500 scenarios/minute
- **Retrieval**: <100ms for 1000s of scenarios
- **LLM generation**: ~2-5 seconds (local Mistral 7B)

## Troubleshooting

### "Cannot connect to Ollama"

1. Install Ollama: https://ollama.ai
2. Pull model: `ollama pull mistral:7b-instruct`
3. Verify service is running

### Slow embedding generation

- Use smaller model: `paraphrase-MiniLM-L3-v2`
- Reduce batch size in config

### Out of memory

- Use 4-bit quantized model
- Reduce embedding model size
- Process scenarios in smaller batches

## Advanced Usage

### Custom Embeddings
```python
from phase2_rag_agent_rl.rag_system.embeddings_generator import EmbeddingsGenerator

# Use different model
embeddings_gen = EmbeddingsGenerator(model_name='all-mpnet-base-v2')
```

### Filter Retrieval
```python
# Retrieve only scenarios with specific criteria
results = retriever.retrieve_by_criteria(
    num_flights=50,
    aircraft_types=['B777'],
    min_delay=10.0,
    n_results=5
)
```

### Batch Processing
```python
questions = [
    "Best practices for peak hour operations?",
    "How to handle equipment failures?",
    "Optimize narrow-body turnaround?"
]

results = rag_engine.batch_query(questions)
```

## Next Steps

1. **Try interactive mode**: `python demo_