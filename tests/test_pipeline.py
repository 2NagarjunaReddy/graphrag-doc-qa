import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graphrag import GraphRAGPipeline, HybridRetriever  # noqa: E402

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


def test_build_from_dir_populates_graph_and_vectors():
    pipeline = GraphRAGPipeline()
    pipeline.build_from_dir(DOCS_DIR)

    assert len(pipeline.vector_store.ids) > 0
    assert len(pipeline.graph) > 0


def test_vector_search_finds_relevant_chunk():
    pipeline = GraphRAGPipeline()
    pipeline.build_from_dir(DOCS_DIR)
    retriever = HybridRetriever(pipeline)

    hits = retriever.retrieve("What is Milvus used for?", top_k=2)

    assert len(hits) > 0
    assert any("milvus" in hit.text.lower() for hit in hits)


def test_save_and_load_round_trip(tmp_path):
    pipeline = GraphRAGPipeline()
    pipeline.build_from_dir(DOCS_DIR)
    pipeline.save(tmp_path / "index")

    reloaded = GraphRAGPipeline()
    reloaded.load(tmp_path / "index")

    assert len(reloaded.vector_store.ids) == len(pipeline.vector_store.ids)
    assert len(reloaded.graph) == len(pipeline.graph)
