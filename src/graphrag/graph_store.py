from __future__ import annotations

from pathlib import Path

import networkx as nx


class KnowledgeGraph:
    """In-memory graph store with the same shape as a Neo4j property graph.

    Swap this for a real neo4j.GraphDatabase driver in production; the
    add_entity / add_relation / neighbors interface stays the same.
    """

    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    def add_entity(self, name: str, source_chunk: str) -> None:
        key = name.lower()
        if self.graph.has_node(key):
            self.graph.nodes[key]["chunks"].add(source_chunk)
        else:
            self.graph.add_node(key, label=name, chunks={source_chunk})

    def add_relation(self, source: str, relation: str, target: str, source_chunk: str) -> None:
        self.add_entity(source, source_chunk)
        self.add_entity(target, source_chunk)
        self.graph.add_edge(source.lower(), target.lower(), relation=relation, chunk=source_chunk)

    def neighbors(self, entity: str, hops: int = 1) -> set[str]:
        key = entity.lower()
        if key not in self.graph:
            return set()
        found = {key}
        frontier = {key}
        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                next_frontier |= set(self.graph.successors(node))
                next_frontier |= set(self.graph.predecessors(node))
            frontier = next_frontier - found
            found |= next_frontier
        found.discard(key)
        return found

    def chunks_for_entities(self, entities: set[str]) -> set[str]:
        chunk_ids: set[str] = set()
        for entity in entities:
            key = entity.lower()
            if key in self.graph:
                chunk_ids |= self.graph.nodes[key]["chunks"]
        return chunk_ids

    def save(self, path: str | Path) -> None:
        exportable = nx.MultiDiGraph()
        for node, data in self.graph.nodes(data=True):
            exportable.add_node(node, label=data["label"], chunks=",".join(data["chunks"]))
        for u, v, data in self.graph.edges(data=True):
            exportable.add_edge(u, v, relation=data["relation"], chunk=data["chunk"])
        nx.write_graphml(exportable, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeGraph":
        kg = cls()
        loaded = nx.read_graphml(str(path))
        for node, data in loaded.nodes(data=True):
            kg.graph.add_node(node, label=data["label"], chunks=set(data["chunks"].split(",")))
        for u, v, data in loaded.edges(data=True):
            kg.graph.add_edge(u, v, relation=data["relation"], chunk=data["chunk"])
        return kg

    def __len__(self) -> int:
        return self.graph.number_of_nodes()
