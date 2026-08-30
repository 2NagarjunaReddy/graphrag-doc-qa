import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graphrag import GraphRAGPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a GraphRAG index from a folder of documents.")
    parser.add_argument("--docs", required=True, help="folder of .txt/.md/.pdf files")
    parser.add_argument("--out", required=True, help="folder to write the index into")
    args = parser.parse_args()

    pipeline = GraphRAGPipeline()
    pipeline.build_from_dir(args.docs)
    pipeline.save(args.out)

    print(f"indexed {len(pipeline.vector_store.ids)} chunks, {len(pipeline.graph)} graph entities -> {args.out}")


if __name__ == "__main__":
    main()
