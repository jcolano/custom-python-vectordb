# PyVectorDB

A high-performance Python vector database with HNSW indexing, quantization, parallel processing, knowledge graph, and real-time subscriptions.

## Features

- **HNSW Indexing** — Sub-millisecond approximate nearest neighbor search
- **Quantization** — 4-32x memory compression with scalar, binary, and product quantizers
- **Parallel Search** — Multi-core BLAS/GEMM acceleration (67x speedup)
- **Knowledge Graph** — Nodes, edges, traversal, and Cypher-like queries
- **Hybrid Search** — Combined vector similarity + graph relationships
- **REST API** — FastAPI server with WebSocket real-time updates
- **Multiple Embeddings** — OpenAI, Sentence Transformers, Cohere, or custom
- **Persistence** — Save/load to disk with automatic recovery

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       PyVectorDB System                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐                  │
│  │  vectordb_       │    │   quantization   │                  │
│  │  optimized.py    │    │      .py         │                  │
│  │  ──────────────  │    │  ──────────────  │                  │
│  │  • VectorDB      │    │  • Scalar (4x)   │                  │
│  │  • Collection    │    │  • Binary (32x)  │                  │
│  │  • HNSW Index    │    │  • Product (8x)  │                  │
│  │  • Filters       │    │                  │                  │
│  └────────┬─────────┘    └────────┬─────────┘                  │
│           │                       │                             │
│           ▼                       ▼                             │
│  ┌──────────────────────────────────────────┐                  │
│  │           parallel_search.py              │                  │
│  │  ────────────────────────────────────────│                  │
│  │  • ParallelSearchEngine (BLAS/GEMM)      │                  │
│  │  • MemoryMappedVectors (>RAM datasets)   │                  │
│  │  • ConcurrentHNSWSearcher                │                  │
│  └────────────────────┬─────────────────────┘                  │
│                       │                                         │
│           ┌───────────┴───────────┐                            │
│           ▼                       ▼                             │
│  ┌──────────────────┐    ┌──────────────────┐                  │
│  │    graph.py      │    │  hybrid_search   │                  │
│  │  ──────────────  │    │      .py         │                  │
│  │  • GraphDB       │    │  ──────────────  │                  │
│  │  • Nodes/Edges   │    │  Vector + Graph  │                  │
│  │  • Traversal     │    │  Combined Search │                  │
│  └──────────────────┘    └──────────────────┘                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Core dependencies
pip install numpy hnswlib

# REST API (optional)
pip install fastapi uvicorn pydantic

# Embeddings (optional - choose one or more)
pip install sentence-transformers  # Local embeddings
pip install openai                 # OpenAI embeddings
pip install cohere                 # Cohere embeddings
```

## Quick Start

### Basic Vector Database

```python
from vectordb_optimized import VectorDB, Filter
import numpy as np

# Create database
db = VectorDB("./my_database")
collection = db.create_collection("documents", dimensions=384)

# Insert vectors
vector = np.random.randn(384).astype(np.float32)
collection.insert(vector, id="doc1", metadata={"category": "tech", "author": "Alice"})

# Batch insert (faster)
vectors = np.random.randn(1000, 384).astype(np.float32)
ids = [f"doc_{i}" for i in range(1000)]
metadata_list = [{"category": "tech"} for _ in range(1000)]
collection.insert_batch(vectors, ids, metadata_list)

# Search
query = np.random.randn(384).astype(np.float32)
results = collection.search(query, k=10)

for r in results:
    print(f"ID: {r.id}, Score: {r.score:.4f}")

# Filtered search
results = collection.search(
    query, k=10,
    filter=Filter.eq("category", "tech")
)

# Save to disk
db.save()
```

### Memory Compression with Quantization

```python
from quantization import ScalarQuantizer, BinaryQuantizer
import numpy as np

vectors = np.random.randn(100000, 384).astype(np.float32)

# Scalar Quantization: 4x compression, 97%+ recall
sq = ScalarQuantizer(dimensions=384)
sq.train(vectors)
quantized = sq.encode(vectors)

print(f"Original: {vectors.nbytes / 1e6:.1f} MB")
print(f"Quantized: {quantized.nbytes / 1e6:.1f} MB")

# Search with quantized vectors
query = np.random.randn(384).astype(np.float32)
distances = sq.distances_l2(query, quantized)
top_k = np.argpartition(distances, 10)[:10]

# Binary Quantization: 32x compression, ultra-fast hamming distance
bq = BinaryQuantizer(dimensions=384)
bq.train(vectors)
binary = bq.encode(vectors)
distances = bq.distances_hamming(query, binary)
```

### Parallel Processing for Large Datasets

```python
from parallel_search import ParallelSearchEngine, MemoryMappedVectors
import numpy as np

vectors = np.random.randn(1000000, 128).astype(np.float32)
query = np.random.randn(128).astype(np.float32)

# Parallel search with BLAS (67x faster than naive)
engine = ParallelSearchEngine(n_workers=8)
results = engine.search_parallel(query, vectors, k=10, metric="cosine")

# Batch search with GEMM (2x faster for multiple queries)
queries = np.random.randn(100, 128).astype(np.float32)
all_results = engine.search_batch_parallel(queries, vectors, k=10)

# Memory-mapped for datasets larger than RAM
mmap = MemoryMappedVectors("./large_dataset", dimensions=128)
mmap.create(n_vectors=100_000_000)
mmap.append_batch(vectors)
results = mmap.search_parallel(query, k=10, engine=engine)
```

### Knowledge Graph

```python
from graph import GraphDB, NodeBuilder, EdgeBuilder

graph = GraphDB()

# Create nodes
graph.create_node(
    NodeBuilder("user_1")
    .label("User")
    .property("name", "Alice")
    .property("role", "engineer")
    .build()
)

graph.create_node(
    NodeBuilder("doc_1")
    .label("Document")
    .property("title", "Vector DB Guide")
    .build()
)

# Create relationships
graph.create_edge(
    EdgeBuilder("user_1", "doc_1", "AUTHORED")
    .property("date", "2024-01-15")
    .build()
)

# Query neighbors
neighbors = graph.neighbors("user_1", direction="out")
for node in neighbors:
    print(f"Connected to: {node.id}")

# Traverse graph
paths = graph.traverse("user_1", max_depth=3)
```

### Embeddings Integration

```python
from embeddings import get_embedder, EmbeddingCollection
from vectordb_optimized import VectorDB

# Auto-detect embedder (uses OPENAI_API_KEY if set, else local)
embedder = get_embedder()

# Or specify provider
embedder = get_embedder("openai", model="text-embedding-3-small")
embedder = get_embedder("sentence-transformers", model="all-MiniLM-L6-v2")

# Create embedding-aware collection
db = VectorDB("./my_db")
collection = db.create_collection("docs", dimensions=embedder.dimensions)
docs = EmbeddingCollection(collection, embedder)

# Insert with automatic embedding
docs.add("doc1", "Machine learning is fascinating", {"category": "AI"})

# Search with text query
results = docs.search("artificial intelligence", k=5)
```

### REST API Server

```bash
# Start server
uvicorn server:app --reload --port 8000

# Or run directly
python server.py
```

```python
# Client usage
from client import VectorDBClient

client = VectorDBClient("http://localhost:8000")

# Create collection
client.create_collection("docs", dimensions=384)

# Insert
client.insert("docs", vector=[0.1, 0.2, ...], metadata={"title": "Hello"})

# Search
results = client.search("docs", vector=[0.1, 0.2, ...], k=10)
```

## Module Reference

| Module | Description |
|--------|-------------|
| `vectordb_optimized.py` | Main vector database with HNSW indexing |
| `quantization.py` | Scalar, binary, and product quantization |
| `parallel_search.py` | Multi-core search engine and memory-mapped vectors |
| `graph.py` | Knowledge graph with nodes, edges, and traversal |
| `hybrid_search.py` | Combined vector + graph search |
| `embeddings.py` | OpenAI, Sentence Transformers, Cohere integration |
| `server.py` | FastAPI REST server |
| `client.py` | Python HTTP client |
| `realtime.py` | WebSocket real-time subscriptions |

### Filter Operations

```python
from vectordb_optimized import Filter

Filter.eq("field", value)       # Equal
Filter.ne("field", value)       # Not equal
Filter.gt("field", value)       # Greater than
Filter.gte("field", value)      # Greater than or equal
Filter.lt("field", value)       # Less than
Filter.lte("field", value)      # Less than or equal
Filter.in_("field", [values])   # In list
Filter.and_([f1, f2])           # AND
Filter.or_([f1, f2])            # OR
```

### Quantization Comparison

| Quantizer | Compression | Recall | Speed | Use Case |
|-----------|-------------|--------|-------|----------|
| `ScalarQuantizer` | 4x | 97%+ | Moderate | Production (balanced) |
| `BinaryQuantizer` | 32x | ~85% | Very Fast | Ultra-fast filtering |
| `ProductQuantizer` | 8-16x | ~90% | Fast | Research/Analytics |

## Benchmarks

Performance on 100K vectors, 128 dimensions:

| Method | Latency | QPS | Memory |
|--------|---------|-----|--------|
| Naive Python | 450 ms | 2 | 48 MB |
| Vectorized BLAS | 6 ms | 167 | 48 MB |
| HNSW | 0.17 ms | 5,773 | 48 MB |
| Scalar Quantized | 6 ms | 167 | 12 MB |
| Binary Quantized | 0.8 ms | 1,250 | 1.5 MB |

Speedup vs Naive (100K vectors):

| Method | Speedup |
|--------|---------|
| BLAS | 89x |
| Parallel Engine | 87x |
| Batch GEMM | 267x |
| HNSW | 2,388x |
| Hybrid | 939x |

### Running Benchmarks

```bash
# Quick benchmark (10K vectors)
python benchmark.py --quick

# Standard benchmark (100K vectors)
python benchmark.py --medium

# Stress test (1M vectors)
python benchmark.py --stress

# Parallel search benchmark
python benchmark_parallel.py

# Quantization benchmark
python benchmark_quantization.py

# Compare implementations
python benchmark_comparison.py
```

## Performance Tuning

### HNSW Parameters

| Parameter | Default | Description | Tradeoff |
|-----------|---------|-------------|----------|
| `M` | 16 | Connections per node | Higher = better recall, more memory |
| `ef_construction` | 200 | Build quality | Higher = better index, slower build |
| `ef_search` | 50 | Search quality | Higher = better recall, slower search |

```python
collection = db.create_collection(
    "docs",
    dimensions=384,
    M=32,
    ef_construction=400,
)
collection.set_ef_search(100)
```

### Memory vs Speed Guidelines

| Dataset Size | Recommendation |
|--------------|----------------|
| < 100K vectors | HNSW only |
| 100K - 1M | HNSW + Scalar quantization |
| 1M - 10M | Memory-mapped + HNSW |
| > 10M | Memory-mapped + Binary quantization + HNSW candidates |

## Integration Patterns

### HNSW + Quantization (Memory Efficient)

```python
from vectordb_optimized import VectorDB
from quantization import ScalarQuantizer

db = VectorDB("./db")
collection = db.create_collection("docs", dimensions=384)
collection.insert_batch(vectors, ids)

# Also keep quantized copy for memory efficiency
sq = ScalarQuantizer(384)
sq.train(vectors)
quantized = sq.encode(vectors)

# Use HNSW for candidate generation, quantized for re-ranking
candidates = collection.search(query, k=100)
```

### HNSW + Graph (Hybrid Search)

```python
from vectordb_optimized import VectorDB
from graph import GraphDB, NodeBuilder, EdgeBuilder

db = VectorDB("./db")
collection = db.create_collection("docs", dimensions=384)
graph = GraphDB()

# Index document in both
collection.insert(vector, id="doc1", metadata={"author": "alice"})
graph.create_node(NodeBuilder("doc1").label("Document").build())
graph.create_edge(EdgeBuilder("doc1", "user_alice", "AUTHORED").build())

# Hybrid search: vector similarity + graph expansion
vector_results = collection.search(query, k=10)
for r in vector_results:
    related = graph.neighbors(r.id, direction="both")
```

## Demo

```bash
# Interactive demo showcasing all features
python demo.py

# RAG application demo
python rag_demo.py

# API reference
python rag_demo.py --api

# RAG example with LLM
python rag_example.py --llm -q "Explain neural networks"
```

## Requirements

```
numpy>=1.24.0
hnswlib>=0.8.0
fastapi>=0.109.0      # REST API
uvicorn>=0.27.0       # ASGI server
pydantic>=2.0.0       # Data validation
websockets>=12.0      # Real-time updates
httpx>=0.26.0         # HTTP client
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
