from __future__ import annotations

from dataclasses import dataclass

import spacy

_KEEP_LABELS = {"PERSON", "ORG", "PRODUCT", "GPE", "WORK_OF_ART", "NORP", "FAC", "EVENT"}


@dataclass
class Relation:
    source: str
    relation: str
    target: str


class EntityExtractor:
    def __init__(self, model: str = "en_core_web_sm") -> None:
        self.nlp = spacy.load(model)

    def extract_entities(self, text: str) -> list[str]:
        doc = self.nlp(text)
        seen = {}
        for ent in doc.ents:
            if ent.label_ in _KEEP_LABELS:
                seen[ent.text.strip().lower()] = ent.text.strip()
        return list(seen.values())

    def extract_relations(self, text: str) -> list[Relation]:
        """Pull simple subject-verb-object triples out of the dependency parse."""
        doc = self.nlp(text)
        relations = []
        for sent in doc.sents:
            for token in sent:
                if token.pos_ != "VERB":
                    continue
                subjects = [c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")]
                objects = [c for c in token.children if c.dep_ in ("dobj", "pobj", "attr")]
                for subj in subjects:
                    for obj in objects:
                        relations.append(
                            Relation(
                                source=_span_text(subj),
                                relation=token.lemma_,
                                target=_span_text(obj),
                            )
                        )
        return relations


def _span_text(token) -> str:
    # walk compound/amod children to capture short noun phrases like "vector database"
    parts = sorted([token, *[c for c in token.children if c.dep_ in ("compound", "amod")]], key=lambda t: t.i)
    return " ".join(t.text for t in parts).strip()
