# FastPyVectorDB Technical Manual

A comprehensive technical reference for the FastPyVectorDB vector database system.

---

## Overview

**FastPyVectorDB** is a high-performance Python vector database designed for semantic search, RAG (Retrieval Augmented Generation) applications, and embedding-based retrieval. It provides a simple, ChromaDB-like API while offering advanced features like HNSW indexing, multiple embedding providers, quantization, knowledge graphs, and hybrid search.

### Key Features

| Feature | Description |
|---------|-------------|
| **Simple API** | ChromaDB-compatible interface for easy adoption |
| **Multiple Embeddings** | Local (Sentence Transformers), OpenAI, Cohere |
| **HNSW Indexing** | Sub-millisecond approximate nearest neighbor search |
| **Metadata Filtering** | Rich filter operations on any metadata field |
| **Quantization** | 4-32x memory compression (Scalar, Binary, Product) |
| **Parallel Search** | Multi-core BLAS/GEMM acceleration |
| **Knowledge Graph** | Property graph with nodes, edges, and traversal |
| **Hybrid Search** | Combined vector similarity + BM25 keyword search |
| **REST API** | FastAPI server with WebSocket real-time updates |
| **Persistence** | Automatic save/load with disk serialization |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Application                          │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│              fastpyvectordb.Client (High-Level API)              │
│         create_collection() | query() | add() | delete()         │
└──────────┬─────────────────────────────────────────┬────────────┘
           │                                         │
┌──────────▼──────────────┐            ┌─────────────▼────────────┐
│  Collection (Wrapper)    │            │      Embedder            │
│  • Auto text→vector      │            │  • OpenAIEmbedder        │
│  • Document storage      │            │  • SentenceTransformer   │
│  • Metadata management   │            │  • CohereEmbedder        │
└──────────┬───────────────┘            │  • MockEmbedder          │
           │                            └──────────────────────────┘
┌──────────▼───────────────────────────────────────────────────────┐
│              vectordb_optimized.Collection (Core Engine)          │
│   • HNSW Index (hnswlib)        • Vector Matrix Cache             │
│   • Batch Operations            • Filter Engine                   │
│   • Disk Serialization          • Thread-safe (RLock)             │
└──────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     Advanced Modules (Optional)                   │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│ Hybrid Search│ Quantization │   Graph DB   │  Parallel Search   │
│  (BM25+Vec)  │  (SQ/BQ/PQ)  │ (Nodes/Edge) │  (Multi-core)      │
└──────────────┴──────────────┴──────────────┴────────────────────┘
```

---

## Module Reference

### Package Structure

```
custom-python-vectordb/
├── fastpyvectordb/              # Public API package
│   ├── __init__.py              # Exports and convenience functions
│   ├── client.py                # High-level Client & Collection API
│   └── py.typed                 # Type hints marker
├── vectordb_optimized.py        # Core database engine
├── embeddings.py                # Embedding providers
├── quantization.py              # Vector compression
├── parallel_search.py           # Multi-core processing
├── hybrid_search.py             # Vector + keyword search
├── graph.py                     # Property graph database
├── realtime.py                  # WebSocket events
├── server.py                    # REST API server
├── client.py                    # HTTP client
├── binary_persistence.py        # Binary serialization
├── examples/                    # Usage examples
└── tests/                       # Unit tests
```

---

## Core Modules

### 1. `fastpyvectordb/client.py` — High-Level API

**Purpose:** User-facing ChromaDB-compatible interface

#### Classes

##### `Client`
Main entry point for database operations.

```python
import fastpyvectordb

client = fastpyvectordb.Client(
    path="./vectordb",              # Storage directory
    embedding_model="all-MiniLM-L6-v2",  # Default model
    embedding_provider="auto"        # auto, sentence-transformers, openai, mock
)
```

| Method | Description |
|--------|-------------|
| `create_collection(name, embedding_model, embedding_provider, metadata, distance_metric)` | Create new collection |
| `get_collection(name, embedding_model, embedding_provider)` | Get existing collection |
| `get_or_create_collection(name, ...)` | Idempotent create/get |
| `list_collections()` | List all collection names |
| `delete_collection(name)` | Remove collection and data |
| `persist()` | Save all data to disk |
| `reset()` | Clear all collections |
| `heartbeat()` | Health check (returns timestamp) |

##### `Collection`
Document collection with automatic embedding.

| Method | Description |
|--------|-------------|
| `add(documents, embeddings, ids, metadatas)` | Insert documents |
| `upsert(documents, embeddings, ids, metadatas)` | Insert or update |
| `update(ids, documents, embeddings, metadatas)` | Modify existing |
| `delete(ids, where)` | Remove by ID or filter |
| `query(query_texts, query_embeddings, n_results, where, include)` | Semantic search |
| `get(ids, where, limit, offset, include)` | Retrieve by ID/filter |
| `peek(limit)` | Sample documents |
| `count` | Property: document count |

##### `QueryResult` / `GetResult`
Result dataclasses returned by query/get operations.

```python
@dataclass
class QueryResult:
    ids: list[list[str]]              # [[id1, id2, ...], ...]
    documents: list[list[str]]        # [[doc1, doc2, ...], ...]
    metadatas: list[list[dict]]       # [[{...}, {...}], ...]
    distances: list[list[float]]      # [[0.1, 0.2, ...], ...]
    embeddings: Optional[list[list[np.ndarray]]]
```

---

### 2. `vectordb_optimized.py` — Core Engine

**Purpose:** Optimized vector storage with HNSW indexing

#### Key Classes

##### `VectorDB`
Multi-collection database manager.

```python
from vectordb_optimized import VectorDB

db = VectorDB("./my_database")
collection = db.create_collection("docs", dimensions=384, metric="cosine")
```

| Method | Description |
|--------|-------------|
| `create_collection(name, dimensions, metric, M, ef_construction)` | Create collection |
| `get_collection(name)` | Get existing collection |
| `delete_collection(name)` | Remove collection |
| `list_collections()` | List collection names |
| `save()` | Persist all collections |

##### `Collection` (Core)
Low-level vector collection with HNSW index.

| Method | Description |
|--------|-------------|
| `insert(vector, id, metadata)` | Insert single vector |
| `insert_batch(vectors, ids, metadata_list)` | Optimized batch insert |
| `upsert(vector, id, metadata)` | Insert or replace |
| `get(id, include_vector)` | Retrieve by ID |
| `get_batch(ids, include_vectors)` | Batch retrieval |
| `delete(id)` / `delete_batch(ids)` | Remove vectors |
| `search(query, k, filter, include_vectors, ef_search)` | HNSW search |
| `search_batch(queries, k, filter, ef_search)` | Batch search |
| `brute_force_search(query, k, filter)` | Exact search |
| `save()` / `_load()` | Persistence |
| `count()` | Vector count |
| `list_ids(limit, offset)` | Enumerate IDs |

##### `Filter`
Composable metadata filter system.

```python
from vectordb_optimized import Filter

# Simple filters
filter = Filter.eq("category", "tech")
filter = Filter.gte("score", 0.8)
filter = Filter.in_("status", ["published", "draft"])
filter = Filter.contains("title", "python")
filter = Filter.regex("name", "^test.*")

# Boolean combinations
filter = Filter.and_([
    Filter.eq("category", "tech"),
    Filter.gte("score", 0.8)
])
filter = Filter.or_([Filter.eq("a", 1), Filter.eq("b", 2)])
filter = Filter.not_(Filter.eq("deleted", True))

# From dict (simple equality)
filter = Filter.from_dict({"category": "tech", "year": 2024})
```

| Operator | Method | Description |
|----------|--------|-------------|
| `=` | `Filter.eq(field, value)` | Equal |
| `!=` | `Filter.ne(field, value)` | Not equal |
| `>` | `Filter.gt(field, value)` | Greater than |
| `>=` | `Filter.gte(field, value)` | Greater or equal |
| `<` | `Filter.lt(field, value)` | Less than |
| `<=` | `Filter.lte(field, value)` | Less or equal |
| `IN` | `Filter.in_(field, values)` | In list |
| `NOT IN` | `Filter.nin(field, values)` | Not in list |
| `CONTAINS` | `Filter.contains(field, substr)` | Substring match |
| `REGEX` | `Filter.regex(field, pattern)` | Regex match |
| `AND` | `Filter.and_(filters)` | All conditions |
| `OR` | `Filter.or_(filters)` | Any condition |
| `NOT` | `Filter.not_(filter)` | Negate |

##### `DistanceMetric`
Supported distance metrics.

```python
from vectordb_optimized import DistanceMetric

DistanceMetric.COSINE       # Cosine similarity (default)
DistanceMetric.EUCLIDEAN    # L2 distance
DistanceMetric.DOT_PRODUCT  # Inner product
```

##### `CollectionConfig`
Collection configuration dataclass.

```python
@dataclass
class CollectionConfig:
    name: str
    dimensions: int
    metric: str = "cosine"
    M: int = 16                    # HNSW connections per node
    ef_construction: int = 200     # Build quality
    ef_search: int = 50            # Search quality
    max_elements: int = 100000     # Initial capacity
```

#### HNSW Parameters

| Parameter | Default | Description | Trade-off |
|-----------|---------|-------------|-----------|
| `M` | 16 | Connections per node | Higher = better recall, more memory |
| `ef_construction` | 200 | Build quality | Higher = better index, slower build |
| `ef_search` | 50 | Search quality | Higher = better recall, slower search |

---

### 3. `embeddings.py` — Embedding Providers

**Purpose:** Generate vector embeddings from text

#### Base Class

```python
class Embedder:
    @property
    def dimensions(self) -> int: ...
    @property
    def model_name(self) -> str: ...
    def embed(self, text: str) -> np.ndarray: ...
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray: ...
```

#### Implementations

##### `SentenceTransformerEmbedder` (Local, No API Key)

```python
from embeddings import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
vector = embedder.embed("Hello world")  # Shape: (384,)
```

| Model | Dimensions | Speed | Quality |
|-------|------------|-------|---------|
| `all-MiniLM-L6-v2` | 384 | Fast | Good |
| `all-mpnet-base-v2` | 768 | Medium | Better |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Fast | Multilingual |

##### `OpenAIEmbedder` (Requires API Key)

```python
from embeddings import OpenAIEmbedder

embedder = OpenAIEmbedder(
    model="text-embedding-3-small",
    api_key="sk-...",  # or OPENAI_API_KEY env var
    dimensions=1536    # Optional: reduce dimensions
)
```

| Model | Dimensions | Notes |
|-------|------------|-------|
| `text-embedding-3-small` | 1536 | Cost-effective |
| `text-embedding-3-large` | 3072 | Higher quality |
| `text-embedding-ada-002` | 1536 | Legacy |

##### `CohereEmbedder` (Requires API Key)

```python
from embeddings import CohereEmbedder

embedder = CohereEmbedder(
    model="embed-english-v3.0",
    api_key="...",  # or COHERE_API_KEY env var
    input_type="search_document"  # or "search_query"
)
```

##### `MockEmbedder` (Testing)

```python
from embeddings import MockEmbedder

embedder = MockEmbedder(dimensions=384)
# Deterministic embeddings from text hash
```

##### `CachedEmbedder` (Development)

```python
from embeddings import CachedEmbedder, OpenAIEmbedder

base = OpenAIEmbedder()
embedder = CachedEmbedder(base, cache_path="./embedding_cache.json")
# Avoids repeated API calls for same text
```

#### Factory Function

```python
from embeddings import get_embedder

# Auto-detect: OpenAI (if key) → SentenceTransformers → Mock
embedder = get_embedder()

# Explicit provider
embedder = get_embedder(provider="openai", model="text-embedding-3-small")
embedder = get_embedder(provider="sentence-transformers", model="all-MiniLM-L6-v2")
embedder = get_embedder(provider="cohere", model="embed-english-v3.0")
embedder = get_embedder(provider="mock", model=None)
```

---

### 4. `quantization.py` — Memory Compression

**Purpose:** Reduce memory footprint for large datasets

#### Quantizers

##### `ScalarQuantizer` — 4x Compression, ~97% Recall

Converts float32 → uint8 using min-max normalization.

```python
from quantization import ScalarQuantizer
import numpy as np

vectors = np.random.randn(100000, 384).astype(np.float32)

sq = ScalarQuantizer(dimensions=384)
sq.train(vectors)  # Learn min/max per dimension

quantized = sq.encode(vectors)  # uint8, 4x smaller
decoded = sq.decode(quantized)  # Approximate reconstruction

# Distance calculation
query = np.random.randn(384).astype(np.float32)
distances = sq.distances_l2(query, quantized)
```

##### `BinaryQuantizer` — 32x Compression, ~85% Recall

Converts float32 → 1-bit binary codes.

```python
from quantization import BinaryQuantizer

bq = BinaryQuantizer(dimensions=384)
bq.train(vectors)

binary = bq.encode(vectors)  # Packed bits
distances = bq.distances_hamming(query, binary)
```

##### `ProductQuantizer` — 8-16x Compression, ~90% Recall

Subspace clustering for approximate distances.

```python
from quantization import ProductQuantizer

pq = ProductQuantizer(dimensions=384, n_subquantizers=8, n_clusters=256)
pq.train(vectors)

codes = pq.encode(vectors)
distances = pq.distances(query, codes)
```

#### Comparison

| Quantizer | Compression | Recall | Speed | Use Case |
|-----------|-------------|--------|-------|----------|
| Scalar | 4x | ~97% | Medium | Production (balanced) |
| Binary | 32x | ~85% | Very Fast | Ultra-fast filtering |
| Product | 8-16x | ~90% | Fast | Research/Analytics |

---

### 5. `hybrid_search.py` — Vector + Keyword Search

**Purpose:** Combine semantic similarity with keyword matching

#### Classes

##### `BM25Index`
Best Matching 25 keyword index.

```python
from hybrid_search import BM25Index

bm25 = BM25Index()
bm25.add_document("doc1", "machine learning is great")
bm25.add_document("doc2", "deep learning uses neural networks")

scores = bm25.search("machine learning", k=10)
# Returns: [("doc1", 0.95), ("doc2", 0.45)]
```

##### `HybridCollection`
Combined vector + keyword search.

```python
from hybrid_search import HybridCollection
from vectordb_optimized import VectorDB
from embeddings import get_embedder

db = VectorDB("./hybrid_db")
base_collection = db.create_collection("docs", dimensions=384)
embedder = get_embedder()

hybrid = HybridCollection(base_collection, embedder)

# Add documents (indexed in both systems)
hybrid.add("doc1", "Machine learning is a subset of AI", {"topic": "AI"})

# Hybrid search
results = hybrid.hybrid_search(
    query_text="artificial intelligence",
    k=10,
    alpha=0.7  # 0.7 vector, 0.3 keyword
)
```

| Alpha | Behavior |
|-------|----------|
| 0.0 | Pure keyword (BM25) |
| 0.5 | Balanced |
| 1.0 | Pure vector similarity |

---

### 6. `graph.py` — Property Graph Database

**Purpose:** Store and query graph relationships

#### Core Classes

##### `Node`
```python
from graph import Node, NodeBuilder

node = NodeBuilder("user_1") \
    .label("User") \
    .property("name", "Alice") \
    .property("role", "engineer") \
    .build()
```

##### `Edge`
```python
from graph import Edge, EdgeBuilder

edge = EdgeBuilder("user_1", "doc_1", "AUTHORED") \
    .property("date", "2024-01-15") \
    .build()
```

##### `GraphDB`
```python
from graph import GraphDB

graph = GraphDB(path="./graph_data")

# Create nodes
graph.create_node(node)

# Create edges
graph.create_edge(edge)

# Query neighbors
neighbors = graph.neighbors("user_1", direction="out")

# Traverse graph
paths = graph.traverse("user_1", max_depth=3)

# Pattern matching (simplified Cypher)
results = graph.match("(u:User)-[:AUTHORED]->(d:Document)")
```

---

### 7. `parallel_search.py` — Multi-Core Processing

**Purpose:** Accelerate search on multi-core systems

#### Classes

##### `ParallelSearchEngine`
```python
from parallel_search import ParallelSearchEngine
import numpy as np

vectors = np.random.randn(1000000, 128).astype(np.float32)
query = np.random.randn(128).astype(np.float32)

engine = ParallelSearchEngine(n_workers=8)

# Single query, parallel distance computation
results = engine.search_parallel(query, vectors, k=10, metric="cosine")

# Batch queries
queries = np.random.randn(100, 128).astype(np.float32)
all_results = engine.search_batch_parallel(queries, vectors, k=10)
```

##### `MemoryMappedVectors`
For datasets larger than RAM.

```python
from parallel_search import MemoryMappedVectors

mmap = MemoryMappedVectors("./large_dataset", dimensions=128)
mmap.create(n_vectors=100_000_000)  # Pre-allocate
mmap.append_batch(vectors)

results = mmap.search_parallel(query, k=10, engine=engine)
```

---

### 8. `server.py` — REST API

**Purpose:** HTTP API for remote access

#### Starting the Server

```bash
# Using uvicorn directly
uvicorn server:app --host 0.0.0.0 --port 8000

# Or run the script
python server.py
```

#### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/collections` | List collections |
| POST | `/collections` | Create collection |
| GET | `/collections/{name}` | Get collection info |
| DELETE | `/collections/{name}` | Delete collection |
| POST | `/collections/{name}/insert` | Insert vector |
| POST | `/collections/{name}/batch_insert` | Batch insert |
| POST | `/collections/{name}/search` | Search |
| POST | `/collections/{name}/batch_search` | Batch search |
| GET | `/collections/{name}/get/{id}` | Get by ID |
| DELETE | `/collections/{name}/delete/{id}` | Delete by ID |

#### Client Usage

```python
from client import VectorDBClient

client = VectorDBClient("http://localhost:8000")

# Create collection
client.create_collection("docs", dimensions=384)

# Insert
client.insert("docs", vector=[0.1, 0.2, ...], id="doc1", metadata={"title": "Hello"})

# Search
results = client.search("docs", vector=[0.1, 0.2, ...], k=10)
```

---

### 9. `realtime.py` — WebSocket Events

**Purpose:** Real-time event streaming

#### Event Types

```python
from realtime import EventType

EventType.INSERT
EventType.UPDATE
EventType.DELETE
EventType.SEARCH
EventType.BATCH_INSERT
EventType.COLLECTION_CREATED
EventType.COLLECTION_DELETED
```

#### Usage with FastAPI

```python
from realtime import RealtimeManager

manager = RealtimeManager()

# Broadcast event
manager.broadcast(Event(
    type=EventType.INSERT,
    collection="docs",
    data={"id": "doc1"}
))

# Subscribe to events
subscription = Subscription(
    collection="docs",
    event_types=[EventType.INSERT, EventType.DELETE]
)
```

---

## Persistence Format

Data is stored in the specified directory with the following structure:

```
vectordb/
├── collection_name/
│   ├── index.bin      # HNSW index (binary, hnswlib format)
│   ├── metadata.json  # Document metadata
│   ├── state.json     # ID-to-label mappings
│   └── config.json    # Collection configuration
```

---

## Performance Benchmarks

### Search Latency (100K vectors, 128 dimensions)

| Method | Latency | QPS | Memory |
|--------|---------|-----|--------|
| Naive Python | 450 ms | 2 | 48 MB |
| Vectorized BLAS | 6 ms | 167 | 48 MB |
| HNSW | 0.17 ms | 5,773 | 48 MB |
| Scalar Quantized | 6 ms | 167 | 12 MB |
| Binary Quantized | 0.8 ms | 1,250 | 1.5 MB |

### Speedup vs Naive

| Method | Speedup |
|--------|---------|
| NumPy BLAS | 89x |
| Parallel Engine | 87x |
| Batch GEMM | 267x |
| HNSW | 2,388x |

---

## Dependencies

### Core (Required)
```
numpy>=1.24.0
hnswlib>=0.8.0
```

### Optional
```
sentence-transformers>=2.2.0  # Local embeddings
openai>=1.0.0                  # OpenAI embeddings
cohere>=4.0.0                  # Cohere embeddings
fastapi>=0.109.0               # REST API
uvicorn>=0.27.0                # ASGI server
pydantic>=2.0.0                # Request validation
websockets>=12.0               # WebSocket support
httpx>=0.26.0                  # HTTP client
networkx>=3.0                  # Graph visualization
```

### Installation

```bash
# Core only
pip install numpy hnswlib

# With local embeddings (recommended)
pip install -e ".[local]"

# With OpenAI
pip install -e ".[openai]"

# With server
pip install -e ".[server]"

# Everything
pip install -e ".[all]"
```

---

## Quick Reference

### Basic Usage

```python
import fastpyvectordb

# Create client
client = fastpyvectordb.Client(path="./my_db")

# Create collection
collection = client.create_collection("documents")

# Add documents
collection.add(
    documents=["Hello world", "Machine learning is great"],
    ids=["doc1", "doc2"],
    metadatas=[{"topic": "greeting"}, {"topic": "AI"}]
)

# Search
results = collection.query("artificial intelligence", n_results=5)

# Print results
for doc, score in zip(results.documents[0], results.distances[0]):
    print(f"{score:.4f}: {doc}")

# Save
client.persist()
```

### With Filters

```python
# Search with filter
results = collection.query(
    query_texts="search term",
    n_results=10,
    where={"topic": "AI"}
)

# Complex filter
from fastpyvectordb import Filter

results = collection.query(
    query_texts="search term",
    n_results=10,
    filter=Filter.and_([
        Filter.eq("category", "tech"),
        Filter.gte("score", 0.8)
    ])
)
```

### Low-Level API

```python
from vectordb_optimized import VectorDB, Filter
import numpy as np

db = VectorDB("./raw_db")
collection = db.create_collection("vectors", dimensions=384)

# Insert raw vectors
vector = np.random.randn(384).astype(np.float32)
collection.insert(vector, id="vec1", metadata={"source": "test"})

# Search
query = np.random.randn(384).astype(np.float32)
results = collection.search(query, k=10, filter=Filter.eq("source", "test"))

db.save()
```

---

## File Summary

| File | Lines | Purpose |
|------|-------|---------|
| `fastpyvectordb/__init__.py` | 133 | Public exports |
| `fastpyvectordb/client.py` | 715 | High-level Client/Collection API |
| `vectordb_optimized.py` | 957 | Core HNSW-based engine |
| `embeddings.py` | 699 | Embedding providers |
| `quantization.py` | 798 | Vector compression |
| `parallel_search.py` | 1173 | Multi-core processing |
| `hybrid_search.py` | 608 | Vector + keyword search |
| `graph.py` | 1243 | Property graph database |
| `realtime.py` | 636 | WebSocket events |
| `server.py` | 462 | REST API (basic) |
| `server_full.py` | 730 | REST API (extended) |
| `client.py` | 350 | HTTP client |
| `binary_persistence.py` | 557 | Binary serialization |

**Total:** ~8,000+ lines of core implementation

---

## Design Patterns

1. **Lazy Loading** — Embedding models loaded on first use
2. **Caching** — Vector matrix cache, embedding cache
3. **Vectorization** — NumPy BLAS for O(1) distance calculations
4. **Batch Operations** — Native HNSW batch queries
5. **Memory Mapping** — Support for > RAM datasets
6. **Thread Safety** — RLock for concurrent access
7. **Factory Pattern** — `get_embedder()` for provider selection
8. **Composable Filters** — Boolean filter combinations
9. **Provider Abstraction** — Pluggable embedding backends

---

*Generated for FastPyVectorDB v0.1.0*
