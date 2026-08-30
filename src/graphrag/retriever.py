from __future__ import annotations

from dataclasses import dataclass

from .pipeline import GraphRAGPipeline


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    via_graph: bool


class HybridRetriever:
    """Vector search for direct hits, then graph traversal to pull in
    related chunks that share an entity but didn't rank high on embedding
    similarity alone."""

    def __init__(self, pipeline: GraphRAGPipeline) -> None:
        self.pipeline = pipeline

    def retrieve(self, query: str, top_k: int = 3, graph_hops: int = 1) -> list[RetrievedChunk]:
        query_vector = self.pipeline.embedder.encode([query])[0]
        vector_hits = self.pipeline.vector_store.search(query_vector, top_k=top_k)

        results = {
            chunk_id: RetrievedChunk(
                chunk_id=chunk_id,
                text=self.pipeline.vector_store.metadata[chunk_id]["text"],
                score=score,
                via_graph=False,
            )
            for chunk_id, score in vector_hits
        }

        query_entities = self.pipeline.extractor.extract_entities(query)
        related_entities: set[str] = set()
        for entity in query_entities:
            related_entities |= self.pipeline.graph.neighbors(entity, hops=graph_hops)
            related_entities.add(entity)

        graph_chunk_ids = self.pipeline.graph.chunks_for_entities(related_entities)
        for chunk_id in graph_chunk_ids:
            if chunk_id in results or chunk_id not in self.pipeline.vector_store.metadata:
                continue
            results[chunk_id] = RetrievedChunk(
                chunk_id=chunk_id,
                text=self.pipeline.vector_store.metadata[chunk_id]["text"],
                score=0.0,
                via_graph=True,
            )

        return sorted(results.values(), key=lambda r: r.score, reverse=True)
