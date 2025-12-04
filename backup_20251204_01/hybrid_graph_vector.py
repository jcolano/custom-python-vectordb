"""
Hybrid Graph + Vector Search Module

Combines vector similarity search with graph traversal for semantic graph queries.
Inspired by RuVector's HybridIndex architecture.

Usage:
    from hybrid_graph_vector import HybridGraphVectorDB

    db = HybridGraphVectorDB(dimensions=384)

    # Add nodes with embeddings
    db.add_node_with_embedding(
        node=NodeBuilder().label("Document").property("title", "AI Guide").build(),
        embedding=[0.1, 0.2, ...]
    )

    # Semantic graph search: find similar nodes + expand through graph
    results = db.semantic_graph_search(
        query_embedding=[...],
        k=10,
        expand_hops=2,
        filter_labels=["Document"]
    )
"""

import numpy as np
import hnswlib
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set
from pathlib import Path
import json
import struct
import pickle

from graph import GraphDB, Node, Edge, Hyperedge, NodeBuilder, EdgeBuilder


# =============================================================================
# Unified ID Registry (RuVector Pattern)
# =============================================================================

class UnifiedIDRegistry:
    """
    Central ID management - single source of truth for all ID mappings.

    Inspired by RuVector's id_to_idx/idx_to_id pattern for efficient
    bidirectional lookups between string IDs and integer indices.
    """

    def __init__(self):
        self._string_to_int: Dict[str, int] = {}
        self._int_to_string: Dict[int, str] = {}
        self._next_id: int = 0
        self._lock = threading.Lock()

    def get_or_create(self, string_id: str) -> int:
        """Get existing int ID or create new one. Thread-safe."""
        if string_id in self._string_to_int:
            return self._string_to_int[string_id]

        with self._lock:
            # Double-check after acquiring lock
            if string_id not in self._string_to_int:
                idx = self._next_id
                self._next_id += 1
                self._string_to_int[string_id] = idx
                self._int_to_string[idx] = string_id
            return self._string_to_int[string_id]

    def get_int(self, string_id: str) -> Optional[int]:
        """Get int ID for string ID, or None if not found."""
        return self._string_to_int.get(string_id)

    def get_string(self, int_id: int) -> Optional[str]:
        """Get string ID for int ID, or None if not found."""
        return self._int_to_string.get(int_id)

    def remove(self, string_id: str) -> bool:
        """Remove an ID from the registry."""
        with self._lock:
            if string_id in self._string_to_int:
                int_id = self._string_to_int[string_id]
                del self._string_to_int[string_id]
                del self._int_to_string[int_id]
                return True
            return False

    def __len__(self) -> int:
        return len(self._string_to_int)

    def to_dict(self) -> dict:
        return {
            "string_to_int": self._string_to_int,
            "next_id": self._next_id
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'UnifiedIDRegistry':
        registry = cls()
        registry._string_to_int = data.get("string_to_int", {})
        registry._int_to_string = {int(v): k for k, v in registry._string_to_int.items()}
        registry._next_id = data.get("next_id", 0)
        return registry


# =============================================================================
# Search Results
# =============================================================================

@dataclass
class GraphVectorSearchResult:
    """Result from hybrid graph+vector search."""
    node_id: str
    node: Node
    vector_score: float           # Similarity score from vector search
    graph_distance: int           # Hops from seed nodes (0 = direct match)
    combined_score: float         # Weighted combination
    path: List[str] = field(default_factory=list)  # Path from seed node


# =============================================================================
# Hybrid Graph + Vector Database
# =============================================================================

class HybridGraphVectorDB:
    """
    Hybrid database combining graph structure with vector similarity search.

    Inspired by RuVector's HybridIndex architecture:
    - Separate vector indexes for nodes and edges
    - Bidirectional ID mapping
    - Combined graph + vector queries

    Features:
    - O(1) vector similarity search via HNSW
    - O(1) graph traversal via adjacency index
    - Semantic graph expansion: find similar nodes, then traverse graph
    - Filter by labels, properties, or custom predicates
    """

    def __init__(self, dimensions: int = 384, path: str = None,
                 metric: str = 'cosine', M: int = 16, ef_construction: int = 200):
        """
        Initialize hybrid database.

        Args:
            dimensions: Vector dimensions
            path: Persistence path (optional)
            metric: Distance metric ('cosine', 'l2', 'ip')
            M: HNSW M parameter
            ef_construction: HNSW ef_construction parameter
        """
        self.dimensions = dimensions
        self.path = Path(path) if path else None
        self.metric = metric
        self.M = M
        self.ef_construction = ef_construction

        if self.path:
            self.path.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()

        # Graph database with multi-index architecture
        graph_path = str(self.path / "graph") if self.path else None
        self.graph = GraphDB(graph_path)

        # Unified ID registry (RuVector pattern)
        self._id_registry = UnifiedIDRegistry()

        # Separate HNSW indexes for nodes and edges (RuVector pattern)
        self._node_index: Optional[hnswlib.Index] = None
        self._edge_index: Optional[hnswlib.Index] = None

        # Embedding storage
        self._node_embeddings: Dict[str, np.ndarray] = {}
        self._edge_embeddings: Dict[str, np.ndarray] = {}

        # Initialize HNSW indexes
        self._init_indexes()

        # Load from disk
        if self.path:
            self._load()

    def _init_indexes(self):
        """Initialize HNSW indexes."""
        space = self.metric if self.metric != 'cosine' else 'cosine'

        self._node_index = hnswlib.Index(space=space, dim=self.dimensions)
        self._node_index.init_index(max_elements=1_000_000,
                                     ef_construction=self.ef_construction,
                                     M=self.M)
        self._node_index.set_ef(50)

        self._edge_index = hnswlib.Index(space=space, dim=self.dimensions)
        self._edge_index.init_index(max_elements=1_000_000,
                                     ef_construction=self.ef_construction,
                                     M=self.M)
        self._edge_index.set_ef(50)

    # =========================================================================
    # Node Operations with Embeddings
    # =========================================================================

    def add_node_with_embedding(self, node: Node, embedding: List[float]) -> str:
        """
        Add a node with its embedding vector.

        Args:
            node: Node object (from NodeBuilder)
            embedding: Vector embedding for the node

        Returns:
            Node ID
        """
        with self._lock:
            # Add to graph
            node_id = self.graph.create_node(node)

            # Register ID and get integer index
            int_idx = self._id_registry.get_or_create(node_id)

            # Store embedding
            embedding_arr = np.asarray(embedding, dtype=np.float32)
            self._node_embeddings[node_id] = embedding_arr

            # Add to HNSW index
            self._node_index.add_items(embedding_arr.reshape(1, -1), [int_idx])

            return node_id

    def add_edge_with_embedding(self, edge: Edge, embedding: List[float]) -> str:
        """Add an edge with its embedding vector."""
        with self._lock:
            edge_id = self.graph.create_edge(edge)

            int_idx = self._id_registry.get_or_create(f"edge_{edge_id}")

            embedding_arr = np.asarray(embedding, dtype=np.float32)
            self._edge_embeddings[edge_id] = embedding_arr

            self._edge_index.add_items(embedding_arr.reshape(1, -1), [int_idx])

            return edge_id

    def get_node_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """Get embedding for a node."""
        return self._node_embeddings.get(node_id)

    def delete_node(self, node_id: str) -> bool:
        """Delete a node and its embedding."""
        with self._lock:
            if self.graph.delete_node(node_id):
                if node_id in self._node_embeddings:
                    del self._node_embeddings[node_id]
                # Note: HNSW doesn't support deletion, marked as deleted internally
                return True
            return False

    # =========================================================================
    # Vector Search
    # =========================================================================

    def vector_search(self, query_embedding: List[float], k: int = 10,
                      filter_labels: List[str] = None,
                      filter_properties: Dict = None) -> List[GraphVectorSearchResult]:
        """
        Pure vector similarity search on nodes.

        Args:
            query_embedding: Query vector
            k: Number of results
            filter_labels: Only return nodes with these labels
            filter_properties: Only return nodes with these properties

        Returns:
            List of search results sorted by similarity
        """
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)

        # Fetch more if filtering
        fetch_k = k * 10 if (filter_labels or filter_properties) else k
        fetch_k = min(fetch_k, len(self._node_embeddings))

        if fetch_k == 0:
            return []

        # HNSW search
        labels, distances = self._node_index.knn_query(query, k=fetch_k)

        results = []
        for i in range(len(labels[0])):
            int_idx = int(labels[0][i])
            node_id = self._id_registry.get_string(int_idx)

            if not node_id:
                continue

            node = self.graph.get_node(node_id)
            if not node:
                continue

            # Apply filters
            if filter_labels and not any(l in node.labels for l in filter_labels):
                continue

            if filter_properties:
                match = all(node.properties.get(k) == v
                           for k, v in filter_properties.items())
                if not match:
                    continue

            # Convert distance to similarity (lower distance = higher similarity)
            similarity = 1.0 - float(distances[0][i]) if self.metric == 'cosine' else -float(distances[0][i])

            results.append(GraphVectorSearchResult(
                node_id=node_id,
                node=node,
                vector_score=similarity,
                graph_distance=0,
                combined_score=similarity,
                path=[node_id]
            ))

            if len(results) >= k:
                break

        return results

    # =========================================================================
    # Semantic Graph Search (RuVector's Hybrid Pattern)
    # =========================================================================

    def semantic_graph_search(
        self,
        query_embedding: List[float],
        k: int = 10,
        expand_hops: int = 2,
        filter_labels: List[str] = None,
        filter_properties: Dict = None,
        edge_types: List[str] = None,
        vector_weight: float = 0.7,
        graph_weight: float = 0.3,
        include_seed_only: bool = False
    ) -> List[GraphVectorSearchResult]:
        """
        Semantic graph search: vector similarity + graph expansion.

        This is the key RuVector-inspired feature that combines:
        1. Vector similarity to find semantically similar "seed" nodes
        2. Graph traversal to expand to related nodes
        3. Combined scoring based on vector similarity and graph distance

        Args:
            query_embedding: Query vector
            k: Number of final results
            expand_hops: How many graph hops to expand (0 = no expansion)
            filter_labels: Only include nodes with these labels
            filter_properties: Only include nodes with these properties
            edge_types: Only traverse edges of these types
            vector_weight: Weight for vector similarity (0-1)
            graph_weight: Weight for graph proximity (0-1)
            include_seed_only: If True, only return seed nodes (no expansion)

        Returns:
            List of results sorted by combined score
        """
        # Step 1: Vector search for seed nodes
        seed_k = k * 2  # Get more seeds for better coverage
        seed_results = self.vector_search(query_embedding, k=seed_k)

        if not seed_results or include_seed_only:
            return seed_results[:k]

        # Step 2: Graph expansion from seed nodes
        seed_ids = {r.node_id for r in seed_results}
        seed_scores = {r.node_id: r.vector_score for r in seed_results}

        expanded_nodes: Dict[str, GraphVectorSearchResult] = {}

        # Add seeds first
        for result in seed_results:
            expanded_nodes[result.node_id] = result

        # BFS expansion
        if expand_hops > 0:
            frontier = set(seed_ids)
            visited = set(seed_ids)

            for hop in range(1, expand_hops + 1):
                new_frontier = set()

                for node_id in frontier:
                    # Get neighbors through edges
                    neighbors = self.graph.neighbors(node_id, direction="both",
                                                     edge_type=edge_types[0] if edge_types and len(edge_types) == 1 else None)

                    for neighbor in neighbors:
                        if neighbor.id not in visited:
                            visited.add(neighbor.id)
                            new_frontier.add(neighbor.id)

                            # Calculate score for expanded node
                            # Use closest seed's score decayed by distance
                            best_seed_score = max(
                                seed_scores.get(s, 0) for s in seed_ids
                            )

                            # Decay score based on graph distance
                            decay = 1.0 / (1.0 + hop)

                            # Combined score
                            combined = vector_weight * best_seed_score * decay + graph_weight * (1.0 / hop)

                            # Find path from nearest seed
                            path = [neighbor.id]  # Simplified path

                            expanded_nodes[neighbor.id] = GraphVectorSearchResult(
                                node_id=neighbor.id,
                                node=neighbor,
                                vector_score=best_seed_score * decay,
                                graph_distance=hop,
                                combined_score=combined,
                                path=path
                            )

                frontier = new_frontier

        # Step 3: Apply filters
        final_results = []
        for node_id, result in expanded_nodes.items():
            node = result.node

            # Filter by labels
            if filter_labels and not any(l in node.labels for l in filter_labels):
                continue

            # Filter by properties
            if filter_properties:
                match = all(node.properties.get(k) == v
                           for k, v in filter_properties.items())
                if not match:
                    continue

            final_results.append(result)

        # Sort by combined score (descending)
        final_results.sort(key=lambda r: r.combined_score, reverse=True)

        return final_results[:k]

    # =========================================================================
    # Graph-First Search (Start with graph, rank by vectors)
    # =========================================================================

    def graph_search_with_reranking(
        self,
        start_node_id: str,
        query_embedding: List[float],
        max_hops: int = 3,
        k: int = 10,
        edge_types: List[str] = None
    ) -> List[GraphVectorSearchResult]:
        """
        Start from a specific node, traverse graph, rerank by vector similarity.

        Useful when you know the starting point but want semantically relevant
        neighbors.
        """
        query = np.asarray(query_embedding, dtype=np.float32)

        # Traverse from start node
        paths = self.graph.traverse(start_node_id, edge_type=edge_types[0] if edge_types else None,
                                    max_depth=max_hops)

        results = []
        seen = set()

        for path in paths:
            end_node = path[-1]
            if end_node.id in seen:
                continue
            seen.add(end_node.id)

            # Calculate vector similarity
            embedding = self._node_embeddings.get(end_node.id)
            if embedding is not None:
                if self.metric == 'cosine':
                    similarity = float(np.dot(query, embedding) /
                                      (np.linalg.norm(query) * np.linalg.norm(embedding)))
                else:
                    similarity = -float(np.linalg.norm(query - embedding))
            else:
                similarity = 0

            results.append(GraphVectorSearchResult(
                node_id=end_node.id,
                node=end_node,
                vector_score=similarity,
                graph_distance=len(path) - 1,
                combined_score=similarity,
                path=[n.id for n in path]
            ))

        # Sort by vector similarity
        results.sort(key=lambda r: r.vector_score, reverse=True)

        return results[:k]

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self):
        """Save all data to disk."""
        if not self.path:
            return

        with self._lock:
            # Save graph (uses its own persistence)
            self.graph.save()

            # Save HNSW indexes
            self._node_index.save_index(str(self.path / "node_index.bin"))
            self._edge_index.save_index(str(self.path / "edge_index.bin"))

            # Save ID registry
            with open(self.path / "id_registry.json", "w") as f:
                json.dump(self._id_registry.to_dict(), f)

            # Save embeddings (binary for efficiency)
            self._save_embeddings_binary()

    def _save_embeddings_binary(self):
        """Save embeddings in binary format (3-5x smaller than JSON)."""
        # Node embeddings
        with open(self.path / "node_embeddings.bin", "wb") as f:
            # Header: count, dimensions
            f.write(struct.pack('II', len(self._node_embeddings), self.dimensions))

            # ID list (pickled for simplicity)
            ids = list(self._node_embeddings.keys())
            id_bytes = pickle.dumps(ids)
            f.write(struct.pack('I', len(id_bytes)))
            f.write(id_bytes)

            # Vectors as contiguous float32
            if self._node_embeddings:
                vectors = np.array([self._node_embeddings[id] for id in ids],
                                  dtype=np.float32)
                vectors.tofile(f)

        # Edge embeddings
        with open(self.path / "edge_embeddings.bin", "wb") as f:
            f.write(struct.pack('II', len(self._edge_embeddings), self.dimensions))

            ids = list(self._edge_embeddings.keys())
            id_bytes = pickle.dumps(ids)
            f.write(struct.pack('I', len(id_bytes)))
            f.write(id_bytes)

            if self._edge_embeddings:
                vectors = np.array([self._edge_embeddings[id] for id in ids],
                                  dtype=np.float32)
                vectors.tofile(f)

    def _load(self):
        """Load all data from disk."""
        # Load ID registry
        registry_path = self.path / "id_registry.json"
        if registry_path.exists():
            with open(registry_path, "r") as f:
                self._id_registry = UnifiedIDRegistry.from_dict(json.load(f))

        # Load HNSW indexes
        node_index_path = self.path / "node_index.bin"
        if node_index_path.exists():
            self._node_index.load_index(str(node_index_path))

        edge_index_path = self.path / "edge_index.bin"
        if edge_index_path.exists():
            self._edge_index.load_index(str(edge_index_path))

        # Load embeddings
        self._load_embeddings_binary()

    def _load_embeddings_binary(self):
        """Load embeddings from binary format."""
        # Node embeddings
        node_emb_path = self.path / "node_embeddings.bin"
        if node_emb_path.exists():
            with open(node_emb_path, "rb") as f:
                count, dims = struct.unpack('II', f.read(8))

                id_len = struct.unpack('I', f.read(4))[0]
                ids = pickle.loads(f.read(id_len))

                if count > 0:
                    vectors = np.fromfile(f, dtype=np.float32).reshape(count, dims)
                    self._node_embeddings = dict(zip(ids, vectors))

        # Edge embeddings
        edge_emb_path = self.path / "edge_embeddings.bin"
        if edge_emb_path.exists():
            with open(edge_emb_path, "rb") as f:
                count, dims = struct.unpack('II', f.read(8))

                id_len = struct.unpack('I', f.read(4))[0]
                ids = pickle.loads(f.read(id_len))

                if count > 0:
                    vectors = np.fromfile(f, dtype=np.float32).reshape(count, dims)
                    self._edge_embeddings = dict(zip(ids, vectors))

    # =========================================================================
    # Statistics
    # =========================================================================

    def stats(self) -> dict:
        """Get database statistics."""
        graph_stats = self.graph.stats()
        return {
            **graph_stats,
            "nodes_with_embeddings": len(self._node_embeddings),
            "edges_with_embeddings": len(self._edge_embeddings),
            "id_registry_size": len(self._id_registry),
            "dimensions": self.dimensions,
            "metric": self.metric
        }


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    import time
    import shutil

    print("=" * 60)
    print("Hybrid Graph + Vector Search Demo")
    print("=" * 60)

    # Clean up
    demo_path = Path("./hybrid_graph_demo")
    if demo_path.exists():
        shutil.rmtree(demo_path)

    # Create database
    db = HybridGraphVectorDB(dimensions=128, path=str(demo_path))

    # Generate sample data
    np.random.seed(42)

    print("\nCreating knowledge graph with embeddings...")

    # Create document nodes
    docs = []
    for i, (title, category) in enumerate([
        ("Machine Learning Basics", "AI"),
        ("Deep Neural Networks", "AI"),
        ("Natural Language Processing", "NLP"),
        ("Computer Vision Guide", "CV"),
        ("Reinforcement Learning", "AI"),
        ("Graph Neural Networks", "AI"),
        ("Transformer Architecture", "NLP"),
        ("Object Detection Methods", "CV"),
    ]):
        node = NodeBuilder(f"doc_{i}").label("Document").label(category) \
            .property("title", title).property("category", category).build()

        # Random embedding (in practice, use real embeddings)
        embedding = np.random.randn(128).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        doc_id = db.add_node_with_embedding(node, embedding.tolist())
        docs.append(doc_id)

    # Create topic nodes
    topics = []
    for topic_name in ["AI", "NLP", "CV", "Deep Learning"]:
        node = NodeBuilder(f"topic_{topic_name}").label("Topic") \
            .property("name", topic_name).build()

        embedding = np.random.randn(128).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        topic_id = db.add_node_with_embedding(node, embedding.tolist())
        topics.append(topic_id)

    # Create relationships
    print("Creating relationships...")

    # Documents -> Topics (BELONGS_TO)
    relationships = [
        (docs[0], topics[0]), (docs[1], topics[0]), (docs[1], topics[3]),
        (docs[2], topics[1]), (docs[3], topics[2]), (docs[4], topics[0]),
        (docs[5], topics[0]), (docs[5], topics[3]), (docs[6], topics[1]),
        (docs[7], topics[2])
    ]

    for doc_id, topic_id in relationships:
        edge = EdgeBuilder(doc_id, topic_id, "BELONGS_TO").build()
        db.graph.create_edge(edge)

    # Document citations (CITES)
    citations = [
        (docs[1], docs[0]),  # Deep NN cites ML Basics
        (docs[5], docs[1]),  # GNN cites Deep NN
        (docs[6], docs[2]),  # Transformer cites NLP
        (docs[4], docs[0]),  # RL cites ML Basics
    ]

    for from_id, to_id in citations:
        edge = EdgeBuilder(from_id, to_id, "CITES").build()
        db.graph.create_edge(edge)

    print(f"\nStats: {db.stats()}")

    # Query
    query_embedding = np.random.randn(128).astype(np.float32)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # 1. Pure vector search
    print("\n--- Pure Vector Search ---")
    results = db.vector_search(query_embedding.tolist(), k=3)
    for r in results:
        print(f"  {r.node_id}: score={r.vector_score:.4f}, title={r.node.properties.get('title', r.node_id)}")

    # 2. Semantic graph search with expansion
    print("\n--- Semantic Graph Search (2 hops expansion) ---")
    results = db.semantic_graph_search(
        query_embedding.tolist(),
        k=5,
        expand_hops=2,
        filter_labels=["Document"]
    )
    for r in results:
        print(f"  {r.node_id}: combined={r.combined_score:.4f}, vec={r.vector_score:.4f}, hops={r.graph_distance}")
        print(f"         title={r.node.properties.get('title', r.node_id)}")

    # 3. Graph search with reranking
    print("\n--- Graph Search from 'doc_0' with Vector Reranking ---")
    results = db.graph_search_with_reranking(
        docs[0],  # Start from "Machine Learning Basics"
        query_embedding.tolist(),
        max_hops=2,
        k=5
    )
    for r in results:
        print(f"  {r.node_id}: vec_score={r.vector_score:.4f}, hops={r.graph_distance}")
        print(f"         title={r.node.properties.get('title', r.node_id)}")
        print(f"         path: {' -> '.join(r.path)}")

    # Save and verify
    print("\n--- Persistence Test ---")
    db.save()
    print("Saved to disk")

    # Reload
    db2 = HybridGraphVectorDB(dimensions=128, path=str(demo_path))
    print(f"Reloaded. Stats: {db2.stats()}")

    # Verify search works after reload
    results = db2.vector_search(query_embedding.tolist(), k=3)
    print(f"Search after reload returned {len(results)} results")

    print("\n" + "=" * 60)
    print("Demo complete!")
