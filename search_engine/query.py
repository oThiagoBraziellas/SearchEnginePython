"""Query language.

    cache eviction          ranked OR: documents with any term, best match first
    +cache +eviction        both terms are required
    "least recently used"   exact phrase (required)
    -redis                  documents containing the term are excluded
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple

from .tokenizer import Tokenizer

_PART = re.compile(r'([+-]?)"([^"]*)"|([+-]?)(\S+)')

# A phrase is a list of (term, offset) where offset is the distance from the first term.
Phrase = List[Tuple[str, int]]


@dataclass
class Query:
    optional: List[str] = field(default_factory=list)
    required: List[str] = field(default_factory=list)
    excluded: List[str] = field(default_factory=list)
    phrases: List[Phrase] = field(default_factory=list)
    excluded_phrases: List[Phrase] = field(default_factory=list)

    @property
    def scoring_terms(self) -> List[str]:
        seen, out = set(), []
        for term in self.optional + self.required + [t for p in self.phrases for t, _ in p]:
            if term not in seen:
                seen.add(term)
                out.append(term)
        return out

    @property
    def has_positive_clause(self) -> bool:
        return bool(self.optional or self.required or self.phrases)


def _as_phrase(tokenizer: Tokenizer, text: str) -> Phrase:
    tokens = tokenizer.tokens(text)
    if not tokens:
        return []
    first = tokens[0].position
    return [(t.term, t.position - first) for t in tokens]


def parse_query(text: str, tokenizer: Tokenizer) -> Query:
    q = Query()
    for match in _PART.finditer(text):
        if match.group(2) is not None:
            sign, body, quoted = match.group(1), match.group(2), True
        else:
            sign, body, quoted = match.group(3), match.group(4), False
        phrase = _as_phrase(tokenizer, body)
        if not phrase:
            continue  # only stop words or punctuation
        if len(phrase) == 1:
            term = phrase[0][0]
            if sign == "-":
                q.excluded.append(term)
            elif sign == "+" or quoted:
                q.required.append(term)
            else:
                q.optional.append(term)
        elif sign == "-":
            q.excluded_phrases.append(phrase)
        else:
            q.phrases.append(phrase)  # multi-word quotes (or hyphenated words) are phrases
    return q
