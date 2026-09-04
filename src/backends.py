from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import dijkstra
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


@dataclass
class RetrievalResult:
    method: str
    chunk_ids: list[int]
    contexts: list[str]
    answer_proxy: str
    latency_ms: float
    context_words: int
    trace: dict

    def to_dict(self) -> dict:
        return asdict(self)


class OfflineGraphIndex:
    """Deterministic retrieval-layer proxy for protocol development.

    It is intentionally named a proxy: it validates the adaptive routing protocol
    without claiming to reproduce the LLM-based entity extraction used by the
    official LightRAG and PathRAG systems.
    """

    METHODS = ("vector", "light_proxy", "path_proxy")

    def __init__(
        self,
        chunks: list[str],
        vector_top_k: int = 3,
        light_top_k: int = 5,
        path_top_k: int = 7,
        graph_neighbors: int = 8,
        graph_similarity_floor: float = 0.05,
        adjacent_edge_weight: float = 0.18,
    ) -> None:
        self.chunks = chunks
        self.top_k = {
            "vector": vector_top_k,
            "light_proxy": light_top_k,
            "path_proxy": path_top_k,
        }
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=60_000,
            sublinear_tf=True,
        )
        self.chunk_matrix = self.vectorizer.fit_transform(chunks)
        self.graph, self.transition, self.cost_graph = self._build_graph(
            graph_neighbors, graph_similarity_floor, adjacent_edge_weight
        )

    def _build_graph(
        self,
        neighbors: int,
        similarity_floor: float,
        adjacent_weight: float,
    ) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
        n_chunks = len(self.chunks)
        n_neighbors = min(neighbors + 1, n_chunks)
        model = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        ).fit(self.chunk_matrix)
        distances, indices = model.kneighbors(self.chunk_matrix)
        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []
        for row in range(n_chunks):
            for distance, col in zip(distances[row, 1:], indices[row, 1:]):
                similarity = max(0.0, 1.0 - float(distance))
                if similarity > similarity_floor:
                    rows.append(row)
                    cols.append(int(col))
                    values.append(similarity)
        for row in range(n_chunks - 1):
            rows.extend((row, row + 1))
            cols.extend((row + 1, row))
            values.extend((adjacent_weight, adjacent_weight))
        graph = csr_matrix((values, (rows, cols)), shape=(n_chunks, n_chunks))
        graph = graph.maximum(graph.T).tocsr()
        graph.setdiag(0)
        graph.eliminate_zeros()
        degrees = np.asarray(graph.sum(axis=1)).ravel()
        transition = (diags(1.0 / np.maximum(degrees, 1e-12)) @ graph).tocsr()
        cost_graph = graph.copy()
        cost_graph.data = 1.0 / np.maximum(cost_graph.data, 1e-6)
        return graph, transition, cost_graph

    @staticmethod
    def _top_indices(scores: np.ndarray, k: int) -> np.ndarray:
        k = min(k, len(scores))
        candidates = np.argpartition(-scores, k - 1)[:k]
        return candidates[np.argsort(-scores[candidates])]

    def _query_similarity(self, query: str) -> np.ndarray:
        query_vector = self.vectorizer.transform([query])
        return (self.chunk_matrix @ query_vector.T).toarray().ravel()

    def _personalized_rank(self, similarity: np.ndarray) -> np.ndarray:
        seeds = self._top_indices(similarity, min(8, len(similarity)))
        personalization = np.zeros(len(self.chunks), dtype=float)
        personalization[seeds] = np.maximum(similarity[seeds], 1e-9)
        personalization /= personalization.sum()
        rank = personalization.copy()
        for _ in range(12):
            rank = 0.35 * personalization + 0.65 * (self.transition.T @ rank)
        return rank / (rank.max() + 1e-12)

    def _path_flow(self, query: str, similarity: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
        clauses = [
            part.strip()
            for part in re.split(
                r"\b(?:and|with|while|versus|compared to|because|if)\b|[,;?]",
                query.lower(),
            )
            if len(part.split()) >= 2
        ]
        seeds: list[int] = []
        for clause in clauses[:4]:
            clause_vector = self.vectorizer.transform([clause])
            clause_similarity = (self.chunk_matrix @ clause_vector.T).toarray().ravel()
            seed = int(np.argmax(clause_similarity))
            if seed not in seeds:
                seeds.append(seed)
        for seed in self._top_indices(similarity, min(4, len(similarity))):
            if int(seed) not in seeds:
                seeds.append(int(seed))
            if len(seeds) >= 4:
                break

        flow = np.zeros(len(self.chunks), dtype=float)
        paths: list[list[int]] = []
        if not seeds:
            return flow, paths
        distances, predecessors = dijkstra(
            self.cost_graph,
            directed=False,
            indices=np.asarray(seeds),
            return_predecessors=True,
            limit=12,
        )
        for source_position, source in enumerate(seeds):
            flow[source] += 1
            for target_position in range(source_position + 1, len(seeds)):
                target = seeds[target_position]
                if not np.isfinite(distances[source_position, target]):
                    continue
                current = target
                path = [current]
                for _ in range(12):
                    if current == source:
                        break
                    current = int(predecessors[source_position, current])
                    if current < 0:
                        path = []
                        break
                    path.append(current)
                if path and path[-1] == source:
                    path.reverse()
                    paths.append(path)
                    for node in path:
                        flow[node] += 1
        if flow.max() > 0:
            flow /= flow.max()
        return flow, paths

    def _extractive_answer(self, query: str, contexts: list[str]) -> str:
        sentences: list[str] = []
        for context in contexts:
            sentences.extend(
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+", context)
                if len(sentence.split()) >= 5
            )
        if not sentences:
            return contexts[0] if contexts else ""
        sentence_matrix = self.vectorizer.transform(sentences)
        query_vector = self.vectorizer.transform([query])
        scores = (sentence_matrix @ query_vector.T).toarray().ravel()
        chosen = self._top_indices(scores, min(2, len(sentences)))
        return " ".join(sentences[int(index)] for index in chosen)

    def retrieve(self, query: str, method: str) -> RetrievalResult:
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}")
        started = time.perf_counter()
        similarity = self._query_similarity(query)
        trace: dict = {}
        if method == "vector":
            selected = self._top_indices(similarity, self.top_k[method])
            trace = {"strategy": "direct_tfidf", "seed_chunks": selected.tolist()}
        else:
            rank = self._personalized_rank(similarity)
            if method == "light_proxy":
                score = 0.55 * similarity + 0.45 * rank
                selected = self._top_indices(score, self.top_k[method])
                trace = {
                    "strategy": "one_hop_graph_diffusion",
                    "seed_chunks": self._top_indices(similarity, 8).tolist(),
                }
            else:
                flow, paths = self._path_flow(query, similarity)
                score = 0.45 * similarity + 0.25 * rank + 0.30 * flow
                selected = self._top_indices(score, self.top_k[method])
                trace = {
                    "strategy": "relation_path_proxy",
                    "paths": paths,
                }
        chunk_ids = [int(index) for index in selected]
        contexts = [self.chunks[index] for index in chunk_ids]
        answer_proxy = self._extractive_answer(query, contexts)
        latency_ms = (time.perf_counter() - started) * 1000
        return RetrievalResult(
            method=method,
            chunk_ids=chunk_ids,
            contexts=contexts,
            answer_proxy=answer_proxy,
            latency_ms=latency_ms,
            context_words=sum(len(context.split()) for context in contexts),
            trace=trace,
        )

