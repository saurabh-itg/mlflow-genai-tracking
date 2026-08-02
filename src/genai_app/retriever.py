"""Zero-dependency TF-IDF retriever.

Real systems put Qdrant / pgvector here. The point of this file is that chunking and
top_k are *tunable knobs*, so the MLflow sweep has something real to compare.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"\w+", text) if len(t) > 2]


def chunk_document(doc_id: str, text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_size - overlap)
    chunks = []
    for i in range(0, len(words), step):
        window = words[i : i + chunk_size]
        if len(window) < 20 and chunks:
            break
        chunks.append(Chunk(doc_id, f"{doc_id}#{len(chunks)}", " ".join(window)))
    return chunks


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._tf: list[Counter] = [Counter(_tokenize(c.text)) for c in chunks]
        df: Counter = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(chunks) or 1
        self._idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self._norms = [self._norm(tf) for tf in self._tf]

    def _norm(self, tf: Counter) -> float:
        return math.sqrt(sum((v * self._idf.get(t, 0.0)) ** 2 for t, v in tf.items())) or 1.0

    def search(self, query: str, top_k: int = 4) -> list[tuple[Chunk, float]]:
        q_tf = Counter(_tokenize(query))
        q_norm = self._norm(q_tf)
        scores = []
        for i, tf in enumerate(self._tf):
            dot = sum(qv * self._idf.get(t, 0.0) * tf.get(t, 0) * self._idf.get(t, 0.0)
                      for t, qv in q_tf.items() if t in tf)
            if dot:
                scores.append((self.chunks[i], dot / (q_norm * self._norms[i])))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    @classmethod
    def from_corpus(cls, corpus_dir: str | Path, chunk_size: int = 120, overlap: int = 20):
        chunks: list[Chunk] = []
        for path in sorted(Path(corpus_dir).glob("*.md")):
            chunks.extend(chunk_document(path.stem, path.read_text(encoding="utf-8"), chunk_size, overlap))
        return cls(chunks)
