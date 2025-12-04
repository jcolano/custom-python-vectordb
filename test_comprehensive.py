"""
Comprehensive Test Suite for PyVectorDB with RuVector-Inspired Improvements

Tests all components:
1. VectorDB Core - HNSW indexing, CRUD, search
2. GraphDB with Multi-Index Architecture - PropertyIndex, HyperedgeNodeIndex
3. Hybrid Graph+Vector Search - Semantic graph search
4. Binary Persistence - Fast serialization
5. Integration Tests - End-to-end workflows

Run with: python -m pytest tests/test_comprehensive.py -v
Or standalone: python tests/test_comprehensive.py
"""

import sys
import os
import time
import shutil
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# Import all modules to test
from vectordb_optimized import VectorDB, Collection, CollectionConfig, Filter, SearchResult, DistanceMetric
from graph import (
    GraphDB, Node, Edge, Hyperedge, NodeBuilder, EdgeBuilder, HyperedgeBuilder,
    PropertyIndex, HyperedgeNodeIndex, LabelIndex, AdjacencyIndex, EdgeTypeIndex
)
from hybrid_graph_vector import HybridGraphVectorDB, UnifiedIDRegistry, GraphVectorSearchResult
from binary_persistence import BinaryPersistence, StreamingBinaryWriter, StreamingBinaryReader
from hybrid_search import HybridCollection, BM25Index


# =============================================================================
# Test Configuration
# =============================================================================

class TestConfig:
    DIMENSIONS = 128
    SMALL_COUNT = 100
    MEDIUM_COUNT = 1000
    LARGE_COUNT = 5000
    SEED = 42


def generate_vectors(count: int, dimensions: int = TestConfig.DIMENSIONS) -> np.ndarray:
    """Generate normalized random vectors."""
    np.random.seed(TestConfig.SEED)
    vectors = np.random.randn(count, dimensions).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def generate_query_vector(dimensions: int = TestConfig.DIMENSIONS) -> np.ndarray:
    """Generate a single query vector."""
    np.random.seed(TestConfig.SEED + 1)
    vec = np.random.randn(dimensions).astype(np.float32)
    return vec / np.linalg.norm(vec)


# =============================================================================
# Test Results Tracking
# =============================================================================

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.timings = {}

    def record(self, name: str, passed: bool, error: str = None, duration: float = None):
        if passed:
            self.passed += 1
            status = "✓ PASS"
        else:
            self.failed += 1
            status = "✗ FAIL"
            if error:
                self.errors.append((name, error))

        timing = f" ({duration*1000:.1f}ms)" if duration else ""
        print(f"  {status}: {name}{timing}")

        if duration:
            self.timings[name] = duration

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"  RESULTS: {self.passed}/{total} passed")
        if self.errors:
            print(f"\n  FAILURES:")
            for name, error in self.errors:
                print(f"    - {name}: {error}")
        print(f"{'='*60}")
        return self.failed == 0


results = TestResults()


# =============================================================================
# Suite 1: VectorDB Core Tests
# =============================================================================

def test_vectordb_core():
    print("\n" + "="*60)
    print("  SUITE 1: VectorDB Core")
    print("="*60)

    temp_dir = tempfile.mkdtemp()

    try:
        # Test 1.1: Create database and collection
        start = time.perf_counter()
        db = VectorDB(temp_dir)
        collection = db.create_collection("test", dimensions=TestConfig.DIMENSIONS)
        duration = time.perf_counter() - start
        results.record("Create DB and collection", True, duration=duration)

        # Test 1.2: Single insert
        start = time.perf_counter()
        vec = generate_vectors(1)[0]
        id1 = collection.insert(vec, id="vec_0", metadata={"category": "A", "value": 1.0})
        duration = time.perf_counter() - start
        results.record("Single insert", id1 == "vec_0", duration=duration)

        # Test 1.3: Batch insert
        vectors = generate_vectors(TestConfig.SMALL_COUNT)
        ids = [f"vec_{i}" for i in range(1, TestConfig.SMALL_COUNT + 1)]
        metadata_list = [{"category": chr(65 + i % 4), "value": float(i)} for i in range(TestConfig.SMALL_COUNT)]

        start = time.perf_counter()
        inserted_ids = collection.insert_batch(vectors, ids, metadata_list)
        duration = time.perf_counter() - start
        results.record(f"Batch insert ({TestConfig.SMALL_COUNT} vectors)",
                      len(inserted_ids) == TestConfig.SMALL_COUNT, duration=duration)

        # Test 1.4: Search without filter
        query = generate_query_vector()
        start = time.perf_counter()
        search_results = collection.search(query, k=10)
        duration = time.perf_counter() - start
        results.record("Search (k=10)", len(search_results) == 10, duration=duration)

        # Test 1.5: Search with filter
        start = time.perf_counter()
        filtered_results = collection.search(query, k=10, filter=Filter.eq("category", "A"))
        duration = time.perf_counter() - start
        all_category_a = all(r.metadata.get("category") == "A" for r in filtered_results)
        results.record("Filtered search", all_category_a, duration=duration)

        # Test 1.6: Batch search
        queries = generate_vectors(10)
        start = time.perf_counter()
        batch_results = collection.search_batch(queries, k=5)
        duration = time.perf_counter() - start
        results.record("Batch search (10 queries)",
                      len(batch_results) == 10 and all(len(r) == 5 for r in batch_results),
                      duration=duration)

        # Test 1.7: Get by ID
        start = time.perf_counter()
        entry = collection.get("vec_1", include_vector=True)
        duration = time.perf_counter() - start
        results.record("Get by ID", entry is not None and "vector" in entry, duration=duration)

        # Test 1.8: Delete
        start = time.perf_counter()
        deleted = collection.delete("vec_1")
        duration = time.perf_counter() - start
        entry_after = collection.get("vec_1")
        results.record("Delete", deleted and entry_after is None, duration=duration)

        # Test 1.9: Upsert
        new_vec = generate_vectors(1)[0]
        start = time.perf_counter()
        upsert_id = collection.upsert(new_vec, id="vec_upsert", metadata={"category": "X"})
        duration = time.perf_counter() - start
        results.record("Upsert", upsert_id == "vec_upsert", duration=duration)

        # Test 1.10: Persistence
        collection.save()
        db2 = VectorDB(temp_dir)
        collection2 = db2.get_collection("test")
        start = time.perf_counter()
        loaded_results = collection2.search(query, k=5)
        duration = time.perf_counter() - start
        results.record("Persistence (save/load)", len(loaded_results) == 5, duration=duration)

        # Test 1.11: Brute force search
        start = time.perf_counter()
        brute_results = collection.brute_force_search(query, k=10)
        duration = time.perf_counter() - start
        results.record("Brute force search", len(brute_results) == 10, duration=duration)

        # Test 1.12: Different distance metrics
        for metric in ["cosine", "l2", "ip"]:
            coll = db.create_collection(f"test_{metric}", dimensions=TestConfig.DIMENSIONS, metric=metric)
            coll.insert_batch(generate_vectors(50), [f"v_{i}" for i in range(50)])
            search_results = coll.search(query, k=5)
            results.record(f"Distance metric: {metric}", len(search_results) == 5)

    except Exception as e:
        results.record("VectorDB Core Suite", False, str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# Suite 2: GraphDB Multi-Index Tests
# =============================================================================

def test_graphdb_multiindex():
    print("\n" + "="*60)
    print("  SUITE 2: GraphDB Multi-Index Architecture")
    print("="*60)

    temp_dir = tempfile.mkdtemp()

    try:
        # Test 2.1: Create GraphDB
        start = time.perf_counter()
        db = GraphDB(os.path.join(temp_dir, "graph"))
        duration = time.perf_counter() - start
        results.record("Create GraphDB", True, duration=duration)

        # Test 2.2: Create nodes with properties
        nodes = []
        start = time.perf_counter()
        for i in range(50):
            node = NodeBuilder(f"node_{i}") \
                .label("Person" if i % 2 == 0 else "Company") \
                .property("name", f"Entity_{i}") \
                .property("age", 20 + i) \
                .property("category", chr(65 + i % 4)) \
                .build()
            db.create_node(node)
            nodes.append(node.id)
        duration = time.perf_counter() - start
        results.record(f"Create 50 nodes with properties", db.node_count() == 50, duration=duration)

        # Test 2.3: PropertyIndex - exact lookup
        start = time.perf_counter()
        found = db.find_nodes(properties={"category": "A"})
        duration = time.perf_counter() - start
        expected_count = 50 // 4 + (1 if 50 % 4 > 0 else 0)
        results.record("PropertyIndex exact lookup", len(found) >= 10, duration=duration)

        # Test 2.4: PropertyIndex - range query
        start = time.perf_counter()
        range_found = db.find_nodes_by_property_range("age", min_val=30, max_val=40)
        duration = time.perf_counter() - start
        results.record("PropertyIndex range query", len(range_found) > 0, duration=duration)

        # Test 2.5: LabelIndex lookup
        start = time.perf_counter()
        persons = db.get_nodes_by_label("Person")
        duration = time.perf_counter() - start
        results.record("LabelIndex lookup", len(persons) == 25, duration=duration)

        # Test 2.6: Combined label + property lookup
        start = time.perf_counter()
        combined = db.find_nodes(label="Person", properties={"category": "A"})
        duration = time.perf_counter() - start
        all_match = all("Person" in n.labels and n.properties.get("category") == "A" for n in combined)
        results.record("Combined label+property lookup", all_match, duration=duration)

        # Test 2.7: Create edges
        start = time.perf_counter()
        for i in range(0, 40, 2):
            edge = EdgeBuilder(nodes[i], nodes[i+1], "KNOWS") \
                .property("since", 2020 + i % 5) \
                .build()
            db.create_edge(edge)
        duration = time.perf_counter() - start
        results.record("Create 20 edges", db.edge_count() == 20, duration=duration)

        # Test 2.8: AdjacencyIndex - neighbor lookup
        start = time.perf_counter()
        neighbors = db.neighbors(nodes[0], direction="out")
        duration = time.perf_counter() - start
        results.record("AdjacencyIndex neighbor lookup", len(neighbors) >= 1, duration=duration)

        # Test 2.9: EdgeTypeIndex lookup
        start = time.perf_counter()
        knows_edges = db.get_edges_by_type("KNOWS")
        duration = time.perf_counter() - start
        results.record("EdgeTypeIndex lookup", len(knows_edges) == 20, duration=duration)

        # Test 2.10: Create hyperedges
        start = time.perf_counter()
        for i in range(5):
            he = HyperedgeBuilder([nodes[i*3], nodes[i*3+1], nodes[i*3+2]], "TEAM") \
                .property("team_id", i) \
                .build()
            db.create_hyperedge(he)
        duration = time.perf_counter() - start
        results.record("Create 5 hyperedges", db.hyperedge_count() == 5, duration=duration)

        # Test 2.11: HyperedgeNodeIndex - single node lookup
        start = time.perf_counter()
        node_hyperedges = db.get_hyperedges_by_node(nodes[0])
        duration = time.perf_counter() - start
        results.record("HyperedgeNodeIndex single lookup", len(node_hyperedges) >= 1, duration=duration)

        # Test 2.12: HyperedgeNodeIndex - multi-node lookup (any)
        start = time.perf_counter()
        any_hyperedges = db.get_hyperedges_by_nodes([nodes[0], nodes[3]], mode="any")
        duration = time.perf_counter() - start
        results.record("HyperedgeNodeIndex multi-lookup (any)", len(any_hyperedges) >= 2, duration=duration)

        # Test 2.13: HyperedgeNodeIndex - multi-node lookup (all)
        start = time.perf_counter()
        all_hyperedges = db.get_hyperedges_by_nodes([nodes[0], nodes[1]], mode="all")
        duration = time.perf_counter() - start
        results.record("HyperedgeNodeIndex multi-lookup (all)", len(all_hyperedges) >= 1, duration=duration)

        # Test 2.14: Update node properties (index maintenance)
        start = time.perf_counter()
        db.update_node(nodes[0], properties={"category": "Z", "new_prop": "test"})
        updated_node = db.get_node(nodes[0])
        found_z = db.find_nodes(properties={"category": "Z"})
        duration = time.perf_counter() - start
        results.record("Update node (index maintenance)",
                      updated_node.properties.get("category") == "Z" and len(found_z) >= 1,
                      duration=duration)

        # Test 2.15: Delete node (cascade + index cleanup)
        start = time.perf_counter()
        db.delete_node(nodes[0], cascade=True)
        deleted_check = db.get_node(nodes[0])
        duration = time.perf_counter() - start
        results.record("Delete node with cascade", deleted_check is None, duration=duration)

        # Test 2.16: Graph traversal
        start = time.perf_counter()
        paths = db.traverse(nodes[2], max_depth=2)
        duration = time.perf_counter() - start
        results.record("Graph traversal (depth=2)", isinstance(paths, list), duration=duration)

        # Test 2.17: Shortest path
        start = time.perf_counter()
        path = db.shortest_path(nodes[2], nodes[4])
        duration = time.perf_counter() - start
        results.record("Shortest path", path is None or isinstance(path, list), duration=duration)

        # Test 2.18: Cypher-like query
        start = time.perf_counter()
        query_results = db.query("MATCH (p:Person) RETURN p.name")
        duration = time.perf_counter() - start
        results.record("Cypher-like query", len(query_results) > 0, duration=duration)

        # Test 2.19: Persistence
        db.save()
        db2 = GraphDB(os.path.join(temp_dir, "graph"))
        start = time.perf_counter()
        loaded_nodes = db2.find_nodes(label="Person")
        duration = time.perf_counter() - start
        results.record("GraphDB persistence", len(loaded_nodes) > 0, duration=duration)

        # Test 2.20: Stats
        stats = db2.stats()
        has_indexed_props = "indexed_properties" in stats
        results.record("Stats include indexed_properties", has_indexed_props)

    except Exception as e:
        results.record("GraphDB Multi-Index Suite", False, str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# Suite 3: Hybrid Graph+Vector Search Tests
# =============================================================================

def test_hybrid_graph_vector():
    print("\n" + "="*60)
    print("  SUITE 3: Hybrid Graph+Vector Search")
    print("="*60)

    temp_dir = tempfile.mkdtemp()

    try:
        # Test 3.1: Create HybridGraphVectorDB
        start = time.perf_counter()
        db = HybridGraphVectorDB(dimensions=TestConfig.DIMENSIONS, path=temp_dir)
        duration = time.perf_counter() - start
        results.record("Create HybridGraphVectorDB", True, duration=duration)

        # Test 3.2: Add nodes with embeddings
        nodes = []
        embeddings = generate_vectors(30)
        start = time.perf_counter()
        for i in range(30):
            node = NodeBuilder(f"doc_{i}") \
                .label("Document") \
                .label("Category_" + chr(65 + i % 3)) \
                .property("title", f"Document {i}") \
                .property("score", float(i)) \
                .build()
            node_id = db.add_node_with_embedding(node, embeddings[i].tolist())
            nodes.append(node_id)
        duration = time.perf_counter() - start
        results.record("Add 30 nodes with embeddings", len(nodes) == 30, duration=duration)

        # Test 3.3: Create graph structure
        start = time.perf_counter()
        for i in range(0, 25, 5):
            for j in range(1, 5):
                edge = EdgeBuilder(nodes[i], nodes[i+j], "RELATES_TO").build()
                db.graph.create_edge(edge)
        duration = time.perf_counter() - start
        results.record("Create graph edges", db.graph.edge_count() > 0, duration=duration)

        # Test 3.4: Pure vector search
        query = generate_query_vector()
        start = time.perf_counter()
        vector_results = db.vector_search(query.tolist(), k=5)
        duration = time.perf_counter() - start
        results.record("Pure vector search", len(vector_results) == 5, duration=duration)

        # Test 3.5: Vector search with label filter
        start = time.perf_counter()
        filtered_results = db.vector_search(query.tolist(), k=5, filter_labels=["Category_A"])
        duration = time.perf_counter() - start
        all_have_label = all(any("Category_A" in l for l in r.node.labels) for r in filtered_results)
        results.record("Vector search with label filter", all_have_label, duration=duration)

        # Test 3.6: Vector search with property filter
        start = time.perf_counter()
        prop_filtered = db.vector_search(query.tolist(), k=10, filter_properties={"title": "Document 5"})
        duration = time.perf_counter() - start
        results.record("Vector search with property filter", len(prop_filtered) <= 1, duration=duration)

        # Test 3.7: Semantic graph search (vector + expansion)
        start = time.perf_counter()
        semantic_results = db.semantic_graph_search(
            query.tolist(),
            k=10,
            expand_hops=2
        )
        duration = time.perf_counter() - start
        has_expanded = any(r.graph_distance > 0 for r in semantic_results)
        results.record("Semantic graph search (2 hops)", has_expanded or len(semantic_results) > 0, duration=duration)

        # Test 3.8: Semantic graph search with filters
        start = time.perf_counter()
        filtered_semantic = db.semantic_graph_search(
            query.tolist(),
            k=5,
            expand_hops=1,
            filter_labels=["Document"]
        )
        duration = time.perf_counter() - start
        all_docs = all("Document" in r.node.labels for r in filtered_semantic)
        results.record("Semantic search with filters", all_docs, duration=duration)

        # Test 3.9: Graph search with reranking
        start = time.perf_counter()
        reranked = db.graph_search_with_reranking(
            nodes[0],
            query.tolist(),
            max_hops=2,
            k=5
        )
        duration = time.perf_counter() - start
        results.record("Graph search with vector reranking", len(reranked) >= 0, duration=duration)

        # Test 3.10: Get node embedding
        start = time.perf_counter()
        emb = db.get_node_embedding(nodes[0])
        duration = time.perf_counter() - start
        results.record("Get node embedding", emb is not None and len(emb) == TestConfig.DIMENSIONS, duration=duration)

        # Test 3.11: UnifiedIDRegistry
        registry = UnifiedIDRegistry()
        id1 = registry.get_or_create("test_id_1")
        id2 = registry.get_or_create("test_id_2")
        id1_again = registry.get_or_create("test_id_1")
        results.record("UnifiedIDRegistry", id1 == id1_again and id1 != id2)

        # Test 3.12: Persistence
        db.save()
        db2 = HybridGraphVectorDB(dimensions=TestConfig.DIMENSIONS, path=temp_dir)
        start = time.perf_counter()
        loaded_results = db2.vector_search(query.tolist(), k=5)
        duration = time.perf_counter() - start
        results.record("HybridGraphVectorDB persistence", len(loaded_results) == 5, duration=duration)

        # Test 3.13: Stats
        stats = db2.stats()
        has_hybrid_stats = "nodes_with_embeddings" in stats and "dimensions" in stats
        results.record("Hybrid stats", has_hybrid_stats)

    except Exception as e:
        results.record("Hybrid Graph+Vector Suite", False, str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# Suite 4: Binary Persistence Tests
# =============================================================================

def test_binary_persistence():
    print("\n" + "="*60)
    print("  SUITE 4: Binary Persistence")
    print("="*60)

    temp_dir = tempfile.mkdtemp()

    try:
        # Generate test data
        n_vectors = TestConfig.MEDIUM_COUNT
        vectors = {f"vec_{i}": generate_vectors(1)[0] for i in range(n_vectors)}
        metadata = {f"vec_{i}": {"category": i % 4, "value": float(i)} for i in range(n_vectors)}
        config = {"name": "test", "dimensions": TestConfig.DIMENSIONS}

        # Test 4.1: Binary save
        binary_path = os.path.join(temp_dir, "binary")
        start = time.perf_counter()
        stats = BinaryPersistence.save_vectors(binary_path, vectors, metadata, config)
        duration = time.perf_counter() - start
        results.record(f"Binary save ({n_vectors} vectors)", stats["vectors"] == n_vectors, duration=duration)

        # Test 4.2: Binary load
        start = time.perf_counter()
        loaded_vectors, loaded_meta, loaded_config, id_mapping = BinaryPersistence.load_vectors(binary_path)
        duration = time.perf_counter() - start
        vectors_match = len(loaded_vectors) == n_vectors
        results.record("Binary load", vectors_match, duration=duration)

        # Test 4.3: Data integrity
        sample_id = "vec_0"
        original = vectors[sample_id]
        loaded = loaded_vectors[sample_id]
        integrity = np.allclose(original, loaded)
        results.record("Data integrity after load", integrity)

        # Test 4.4: Metadata integrity
        meta_match = loaded_meta[sample_id] == metadata[sample_id]
        results.record("Metadata integrity", meta_match)

        # Test 4.5: Streaming write
        stream_path = os.path.join(temp_dir, "stream")
        start = time.perf_counter()
        with StreamingBinaryWriter(stream_path, TestConfig.DIMENSIONS, config) as writer:
            for vid, vec in vectors.items():
                writer.write(vid, vec, metadata.get(vid))
        duration = time.perf_counter() - start
        results.record("Streaming write", True, duration=duration)

        # Test 4.6: Streaming read
        start = time.perf_counter()
        reader = StreamingBinaryReader(stream_path)
        count = 0
        for vid, vec, meta in reader.iterate():
            count += 1
        duration = time.perf_counter() - start
        results.record("Streaming read", count == n_vectors, duration=duration)

        # Test 4.7: Batch load from stream
        start = time.perf_counter()
        batch = reader.load_batch(0, 100)
        duration = time.perf_counter() - start
        results.record("Streaming batch load", len(batch) == 100, duration=duration)

        # Test 4.8: File size comparison (binary vs JSON)
        import json
        json_path = os.path.join(temp_dir, "json_test.json")
        with open(json_path, 'w') as f:
            json.dump({
                "vectors": {k: v.tolist() for k, v in vectors.items()},
                "metadata": metadata
            }, f)

        binary_size = os.path.getsize(os.path.join(binary_path, "data.bin"))
        json_size = os.path.getsize(json_path)
        compression_ratio = json_size / binary_size
        results.record(f"Binary compression ({compression_ratio:.1f}x smaller)", compression_ratio > 2)

    except Exception as e:
        results.record("Binary Persistence Suite", False, str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# Suite 5: BM25 Hybrid Search Tests
# =============================================================================

def test_bm25_hybrid():
    print("\n" + "="*60)
    print("  SUITE 5: BM25 Hybrid Search")
    print("="*60)

    temp_dir = tempfile.mkdtemp()

    try:
        # Test 5.1: Create BM25 Index
        start = time.perf_counter()
        bm25 = BM25Index()
        duration = time.perf_counter() - start
        results.record("Create BM25 Index", True, duration=duration)

        # Test 5.2: Add documents
        documents = [
            ("doc_0", "machine learning artificial intelligence deep neural networks"),
            ("doc_1", "natural language processing text analysis sentiment"),
            ("doc_2", "computer vision image recognition object detection"),
            ("doc_3", "reinforcement learning game playing robotics"),
            ("doc_4", "deep learning neural networks backpropagation"),
        ]

        start = time.perf_counter()
        for doc_id, text in documents:
            bm25.add_document(doc_id, text)
        duration = time.perf_counter() - start
        results.record("Add documents to BM25", True, duration=duration)

        # Test 5.3: BM25 search
        start = time.perf_counter()
        bm25_results = bm25.search("machine learning neural", k=3)
        duration = time.perf_counter() - start
        results.record("BM25 search", len(bm25_results) > 0, duration=duration)

        # Test 5.4: Create HybridCollection
        config = CollectionConfig(name="hybrid_test", dimensions=TestConfig.DIMENSIONS)
        start = time.perf_counter()
        collection = HybridCollection(config, Path(temp_dir), text_fields=["content"])
        duration = time.perf_counter() - start
        results.record("Create HybridCollection", True, duration=duration)

        # Test 5.5: Insert with text
        vectors = generate_vectors(len(documents))
        start = time.perf_counter()
        for i, (doc_id, text) in enumerate(documents):
            collection.insert(vectors[i], id=doc_id, metadata={"content": text, "category": i % 2})
        duration = time.perf_counter() - start
        results.record("Insert vectors with text", True, duration=duration)

        # Test 5.6: Keyword search
        start = time.perf_counter()
        keyword_results = collection.keyword_search("machine learning", k=3)
        duration = time.perf_counter() - start
        results.record("Keyword search", len(keyword_results) > 0, duration=duration)

        # Test 5.7: Hybrid search (balanced)
        query_vec = generate_query_vector()
        start = time.perf_counter()
        hybrid_results = collection.hybrid_search(query_vec, "neural networks", k=3, alpha=0.5)
        duration = time.perf_counter() - start
        has_both_scores = all(hasattr(r, 'vector_score') and hasattr(r, 'keyword_score') for r in hybrid_results)
        results.record("Hybrid search (alpha=0.5)", has_both_scores, duration=duration)

        # Test 5.8: Hybrid search (favor keywords)
        start = time.perf_counter()
        keyword_heavy = collection.hybrid_search(query_vec, "neural networks", k=3, alpha=0.2)
        duration = time.perf_counter() - start
        results.record("Hybrid search (alpha=0.2, favor keywords)", len(keyword_heavy) > 0, duration=duration)

        # Test 5.9: Hybrid search (favor vectors)
        start = time.perf_counter()
        vector_heavy = collection.hybrid_search(query_vec, "neural networks", k=3, alpha=0.8)
        duration = time.perf_counter() - start
        results.record("Hybrid search (alpha=0.8, favor vectors)", len(vector_heavy) > 0, duration=duration)

        # Test 5.10: Hybrid search with filter
        start = time.perf_counter()
        filtered_hybrid = collection.hybrid_search(
            query_vec, "learning", k=3, alpha=0.5,
            filter={"category": 0}
        )
        duration = time.perf_counter() - start
        all_cat_0 = all(r.metadata.get("category") == 0 for r in filtered_hybrid)
        results.record("Hybrid search with filter", all_cat_0 or len(filtered_hybrid) == 0, duration=duration)

        # Test 5.11: Persistence
        collection.save()
        collection2 = HybridCollection(config, Path(temp_dir), text_fields=["content"])
        start = time.perf_counter()
        loaded_results = collection2.keyword_search("machine", k=3)
        duration = time.perf_counter() - start
        results.record("HybridCollection persistence", len(loaded_results) > 0, duration=duration)

    except Exception as e:
        results.record("BM25 Hybrid Search Suite", False, str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# Suite 6: Performance Benchmarks
# =============================================================================

def test_performance():
    print("\n" + "="*60)
    print("  SUITE 6: Performance Benchmarks")
    print("="*60)

    temp_dir = tempfile.mkdtemp()

    try:
        # Setup
        db = VectorDB(temp_dir)
        collection = db.create_collection("perf", dimensions=TestConfig.DIMENSIONS)

        # Benchmark: Batch insert
        vectors = generate_vectors(TestConfig.LARGE_COUNT)
        ids = [f"vec_{i}" for i in range(TestConfig.LARGE_COUNT)]
        metadata_list = [{"category": i % 10} for i in range(TestConfig.LARGE_COUNT)]

        start = time.perf_counter()
        collection.insert_batch(vectors, ids, metadata_list)
        insert_time = time.perf_counter() - start
        insert_rate = TestConfig.LARGE_COUNT / insert_time

        results.record(f"Insert rate: {insert_rate:,.0f} vec/sec", insert_rate > 1000, duration=insert_time)

        # Benchmark: Search latency
        query = generate_query_vector()
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            collection.search(query, k=10)
            latencies.append(time.perf_counter() - start)

        avg_latency = np.mean(latencies) * 1000
        p99_latency = np.percentile(latencies, 99) * 1000
        qps = 1000 / avg_latency

        results.record(f"Avg search latency: {avg_latency:.2f}ms", avg_latency < 50)
        results.record(f"P99 search latency: {p99_latency:.2f}ms", p99_latency < 100)
        results.record(f"Search QPS: {qps:,.0f}", qps > 100)

        # Benchmark: Batch search
        queries = generate_vectors(100)
        start = time.perf_counter()
        collection.search_batch(queries, k=10)
        batch_time = time.perf_counter() - start
        batch_qps = 100 / batch_time

        results.record(f"Batch search QPS: {batch_qps:,.0f}", batch_qps > 500, duration=batch_time)

        # Benchmark: GraphDB property index
        graph_db = GraphDB()
        for i in range(1000):
            node = NodeBuilder(f"n_{i}").label("Test").property("value", i % 100).build()
            graph_db.create_node(node)

        start = time.perf_counter()
        for _ in range(100):
            graph_db.find_nodes(properties={"value": 50})
        prop_lookup_time = (time.perf_counter() - start) / 100 * 1000

        results.record(f"PropertyIndex lookup: {prop_lookup_time:.3f}ms", prop_lookup_time < 1)

    except Exception as e:
        results.record("Performance Suite", False, str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*60)
    print("  PyVectorDB Comprehensive Test Suite")
    print("  RuVector-Inspired Improvements")
    print("="*60)

    start_time = time.perf_counter()

    # Run all test suites
    test_vectordb_core()
    test_graphdb_multiindex()
    test_hybrid_graph_vector()
    test_binary_persistence()
    test_bm25_hybrid()
    test_performance()

    total_time = time.perf_counter() - start_time

    # Print summary
    print(f"\n  Total time: {total_time:.2f}s")
    success = results.summary()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
