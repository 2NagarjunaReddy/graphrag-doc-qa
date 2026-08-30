from __future__ import annotations

from pathlib import Path

from .embeddings import Embedder
from .entities import EntityExtractor
from .graph_store import KnowledgeGraph
from .ingest import build_chunks, load_documents
from .vector_store import VectorStore


class GraphRAGPipeline:
    def __init__(self, embedder: Embedder | None = None, extractor: EntityExtractor | None = None) -> None:
        self.embedder = embedder or Embedder()
        self.extractor = extractor or EntityExtractor()
        self.graph = KnowledgeGraph()
        self.vector_store = VectorStore()

    def build_from_dir(self, docs_dir: str | Path) -> None:
        documents = load_documents(docs_dir)
        chunks = build_chunks(documents)
        if not chunks:
            raise ValueError(f"no supported documents found in {docs_dir}")

        for chunk in chunks:
            for relation in self.extractor.extract_relations(chunk.text):
                self.graph.add_relation(relation.source, relation.relation, relation.target, chunk.chunk_id)
            for entity in self.extractor.extract_entities(chunk.text):
                self.graph.add_entity(entity, chunk.chunk_id)

        vectors = self.embedder.encode([c.text for c in chunks])
        self.vector_store.add(
            ids=[c.chunk_id for c in chunks],
            vectors=vectors,
            metadatas=[{"doc_id": c.doc_id, "text": c.text} for c in chunks],
        )

    def save(self, index_dir: str | Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        self.graph.save(index_dir / "graph.graphml")
        self.vector_store.save(index_dir / "vectors")

    def load(self, index_dir: str | Path) -> None:
        index_dir = Path(index_dir)
        self.graph = KnowledgeGraph.load(index_dir / "graph.graphml")
        self.vector_store = VectorStore.load(index_dir / "vectors")
