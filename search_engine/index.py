"""Inverted index with BM25 ranking, phrase matching and on-disk persistence."""

import gzip
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

from .query import Phrase, Query, parse_query
from .tokenizer import DEFAULT_STOPWORDS, Tokenizer, simple_stem

FORMAT_VERSION = 1


@dataclass(frozen=True)
class SearchResult:
    name: str
    score: float
    snippet: str


@dataclass
class _Doc:
    name: str
    length: int  # number of indexed tokens
    text: str


class Index:
    """Positional inverted index.

    ``postings[term][doc_id]`` is the sorted list of token positions of the
    term in the document. Positions make phrase queries possible; term
    frequency is simply the length of that list.

    Ranking is Okapi BM25 (``k1``, ``b``) with the non-negative idf
    ``ln(1 + (N - n + 0.5) / (n + 0.5))``.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, stopwords=DEFAULT_STOPWORDS, stemming: bool = True):
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("k1 must be >= 0 and b within [0, 1]")
        self.k1 = k1
        self.b = b
        self._stopwords = None if stopwords is None else sorted(stopwords)
        self._stemming = stemming
        self.tokenizer = Tokenizer(stopwords, simple_stem if stemming else None)
        self._postings: Dict[str, Dict[int, List[int]]] = {}
        self._docs: Dict[int, _Doc] = {}
        self._ids: Dict[str, int] = {}
        self._next_id = 1
        self._total_length = 0

    # -- indexing --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._docs)

    def __contains__(self, name: str) -> bool:
        return name in self._ids

    @property
    def vocabulary_size(self) -> int:
        return len(self._postings)

    def add(self, name: str, text: str) -> None:
        """Index ``text`` under ``name``; an existing document is replaced."""
        if name in self._ids:
            self.remove(name)
        doc_id = self._next_id
        self._next_id += 1
        tokens = self.tokenizer.tokens(text)
        for token in tokens:
            self._postings.setdefault(token.term, {}).setdefault(doc_id, []).append(token.position)
        self._docs[doc_id] = _Doc(name, len(tokens), text)
        self._ids[name] = doc_id
        self._total_length += len(tokens)

    def remove(self, name: str) -> bool:
        doc_id = self._ids.pop(name, None)
        if doc_id is None:
            return False
        doc = self._docs.pop(doc_id)
        self._total_length -= doc.length
        for term in {t.term for t in self.tokenizer.tokens(doc.text)}:
            docs = self._postings.get(term)
            if docs is not None:
                docs.pop(doc_id, None)
                if not docs:
                    del self._postings[term]
        return True

    # -- searching -------------------------------------------------------------

    def _idf(self, term: str) -> float:
        n = len(self._postings.get(term, ()))
        return math.log(1 + (len(self._docs) - n + 0.5) / (n + 0.5))

    def _bm25(self, term: str, doc_id: int, avg_len: float) -> float:
        tf = len(self._postings[term].get(doc_id, ()))
        if tf == 0:
            return 0.0
        length = self._docs[doc_id].length
        norm = 1 - self.b + self.b * (length / avg_len if avg_len else 1)
        return self._idf(term) * tf * (self.k1 + 1) / (tf + self.k1 * norm)

    def _phrase_docs(self, phrase: Phrase) -> Set[int]:
        """Documents where the terms occur at exactly the given relative offsets."""
        terms = {t for t, _ in phrase}
        if any(t not in self._postings for t in terms):
            return set()
        candidates = set.intersection(*(set(self._postings[t]) for t in terms))
        first_term, _ = phrase[0]
        matches = set()
        for doc_id in candidates:
            for start in self._postings[first_term][doc_id]:
                if all(
                    _contains(self._postings[t][doc_id], start + offset) for t, offset in phrase[1:]
                ):
                    matches.add(doc_id)
                    break
        return matches

    def candidates(self, q: Query) -> Set[int]:
        """Documents that satisfy the boolean structure of the query."""
        required_sets: List[Set[int]] = [set(self._postings.get(t, ())) for t in q.required]
        required_sets += [self._phrase_docs(p) for p in q.phrases]
        if required_sets:
            docs = set.intersection(*required_sets)
        else:
            docs = set()
            for term in q.optional:
                docs |= set(self._postings.get(term, ()))
        for term in q.excluded:
            docs -= set(self._postings.get(term, ()))
        for phrase in q.excluded_phrases:
            docs -= self._phrase_docs(phrase)
        return docs

    def search(self, query: str, limit: int = 10, snippet_width: int = 80) -> List[SearchResult]:
        q = parse_query(query, self.tokenizer)
        if not q.has_positive_clause or not self._docs:
            return []
        avg_len = self._total_length / len(self._docs)
        terms = q.scoring_terms
        scored = []
        for doc_id in self.candidates(q):
            score = sum(self._bm25(t, doc_id, avg_len) for t in terms if t in self._postings)
            scored.append((-score, self._docs[doc_id].name, doc_id))
        scored.sort()  # best score first, ties by name for determinism
        return [
            SearchResult(name, -neg, self._snippet(self._docs[doc_id].text, set(terms), snippet_width))
            for neg, name, doc_id in scored[:limit]
        ]

    def _snippet(self, text: str, terms: Set[str], width: int) -> str:
        tokens = self.tokenizer.tokens(text)
        hit = next((t for t in tokens if t.term in terms), None)
        if hit is None:
            return " ".join(text[:width].split())
        start = max(0, hit.start - width // 2)
        end = min(len(text), hit.end + width // 2)
        piece = " ".join(text[start:end].split())
        return ("..." if start > 0 else "") + piece + ("..." if end < len(text) else "")

    def stats(self) -> Dict[str, float]:
        return {
            "documents": len(self._docs),
            "terms": len(self._postings),
            "postings": sum(len(d) for d in self._postings.values()),
            "avg_doc_length": self._total_length / len(self._docs) if self._docs else 0.0,
        }

    # -- persistence -----------------------------------------------------------

    def save(self, path: str) -> None:
        """Write the index as gzip-compressed JSON, atomically.

        Positions are delta-encoded (``[3, 7, 12]`` is stored as ``[3, 4, 5]``),
        which keeps numbers small and compresses well.
        """
        payload = {
            "version": FORMAT_VERSION,
            "k1": self.k1,
            "b": self.b,
            "stemming": self._stemming,
            "stopwords": self._stopwords,
            "next_id": self._next_id,
            "docs": {str(i): [d.name, d.length, d.text] for i, d in self._docs.items()},
            "postings": {
                term: {str(doc_id): _delta(pos) for doc_id, pos in docs.items()}
                for term, docs in self._postings.items()
            },
        }
        tmp = path + ".tmp"
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as handle:
            json.dump(payload, handle, separators=(",", ":"), ensure_ascii=False)
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "Index":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("version") != FORMAT_VERSION:
            raise ValueError(f"unsupported index format: {payload.get('version')!r}")
        index = cls(
            k1=payload["k1"],
            b=payload["b"],
            stopwords=payload["stopwords"],
            stemming=payload["stemming"],
        )
        index._next_id = payload["next_id"]
        for key, (name, length, text) in payload["docs"].items():
            index._docs[int(key)] = _Doc(name, length, text)
            index._ids[name] = int(key)
            index._total_length += length
        for term, docs in payload["postings"].items():
            index._postings[term] = {int(k): _undelta(v) for k, v in docs.items()}
        return index

    @classmethod
    def load_or_create(cls, path: str, **kwargs) -> "Index":
        return cls.load(path) if os.path.exists(path) else cls(**kwargs)


def _contains(sorted_positions: List[int], value: int) -> bool:
    lo, hi = 0, len(sorted_positions)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_positions[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo < len(sorted_positions) and sorted_positions[lo] == value


def _delta(positions: List[int]) -> List[int]:
    out, previous = [], 0
    for p in positions:
        out.append(p - previous)
        previous = p
    return out


def _undelta(deltas: List[int]) -> List[int]:
    out, total = [], 0
    for d in deltas:
        total += d
        out.append(total)
    return out
