import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graphrag import GraphRAGPipeline, HybridRetriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Query a saved GraphRAG index.")
    parser.add_argument("--index", required=True, help="folder produced by build_index.py")
    parser.add_argument("--q", required=True, help="question to ask")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    pipeline = GraphRAGPipeline()
    pipeline.load(args.index)
    retriever = HybridRetriever(pipeline)

    for hit in retriever.retrieve(args.q, top_k=args.top_k):
        source = "graph" if hit.via_graph else f"vector ({hit.score:.3f})"
        print(f"[{source}] {hit.chunk_id}\n{hit.text}\n")


if __name__ == "__main__":
    main()
