# graphrag-doc-qa

A small GraphRAG pipeline: parse documents, extract entities and relationships into a knowledge graph, embed chunks into a vector index, and retrieve using both signals together.

Plain vector search finds chunks that *sound* like the query. It misses chunks that are *connected* to the query's topic through an entity but use different wording. This project adds a graph traversal step on top of vector search to catch those cases — the same idea behind GraphRAG in production RAG systems.

## How it works

1. **Ingest** — read `.txt` / `.md` / `.pdf` files and split them into overlapping chunks (`src/graphrag/ingest.py`).
2. **Extract** — run spaCy NER and dependency parsing on each chunk to pull out entities and subject-verb-object relations (`src/graphrag/entities.py`).
3. **Graph** — write entities and relations into an in-memory property graph, keyed the same way a Neo4j graph would be (`src/graphrag/graph_store.py`).
4. **Embed** — encode each chunk with `sentence-transformers` and store the vectors in a cosine-similarity index shaped like a Milvus collection (`src/graphrag/embeddings.py`, `src/graphrag/vector_store.py`).
5. **Retrieve** — at query time, run a vector search for direct hits, then extract entities from the query and walk the graph to pull in related chunks the vector search alone would miss (`src/graphrag/retriever.py`).

The graph and vector stores are self-contained (`networkx` + `numpy`) so the project runs with no external services. The interfaces (`add_entity`/`add_relation`/`neighbors`, `add`/`search`) are shaped to match a real `neo4j` driver and a real Milvus client, so swapping either backend in is a matter of implementing the same methods against the real service.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

Build an index from the sample documents:

```bash
python scripts/build_index.py --docs data/sample_docs --out .index
```

Ask a question against it:

```bash
python scripts/query.py --index .index --q "What is Milvus used for?"
python scripts/query.py --index .index --q "How does GraphRAG combine vector search and a graph?"
```

Point `--docs` at your own folder of text/markdown/PDF files to index anything else.

## Tests

```bash
pytest
```

## Why this design

- **No external services required to try it** — the graph and vector store are pure-Python, so there's nothing to provision before you can see it work.
- **Same interface as production infra** — `KnowledgeGraph` and `VectorStore` expose the operations a real Neo4j graph and Milvus collection would, so this doubles as a working sketch of the production architecture rather than a one-off demo.
- **Relation extraction, not just entity tagging** — subject-verb-object triples from the dependency parse give the graph actual edges to walk, not just a bag of disconnected entities.
