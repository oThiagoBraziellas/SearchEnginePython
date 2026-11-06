"""Text analysis: tokenisation, accent folding, stop words and light stemming."""

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, FrozenSet, Iterable, List, Optional

# Small stop word list (English and Portuguese). Positions of removed words are
# preserved so that phrase queries do not match across a removed word.
DEFAULT_STOPWORDS = frozenset(
    """a an and are as at be but by for if in into is it no not of on or such that the their then
    there these they this to was will with
    o os as um uma uns umas de do da dos das em no na nos nas e ou que com por para se ao aos
    """.split()
)

_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def fold(text: str) -> str:
    """Lowercase and strip accents: ``Configuração`` -> ``configuracao``."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def simple_stem(word: str) -> str:
    """Conservative suffix stripper (English plurals, -ing, -ed, trailing e).

    It is deliberately simple: what matters is that the same function is applied
    to documents and queries, so ``caches``, ``cached`` and ``caching`` all
    meet at ``cach``.
    """
    w = word
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 4 and w.endswith("sses"):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    for suffix in ("ing", "ed"):
        if len(w) > len(suffix) + 2 and w.endswith(suffix):
            w = w[: -len(suffix)]
            if len(w) > 2 and w[-1] == w[-2] and w[-1] not in "aeiouls":
                w = w[:-1]  # running -> runn -> run
            break
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
    return w


@dataclass(frozen=True)
class Token:
    term: str
    position: int  # index among ALL words, including removed stop words
    start: int  # character offsets in the original text
    end: int


class Tokenizer:
    def __init__(
        self,
        stopwords: Optional[Iterable[str]] = DEFAULT_STOPWORDS,
        stemmer: Optional[Callable[[str], str]] = simple_stem,
    ):
        self.stopwords: FrozenSet[str] = frozenset(fold(w) for w in (stopwords or ()))
        self.stemmer = stemmer

    def tokens(self, text: str) -> List[Token]:
        out: List[Token] = []
        for position, match in enumerate(_WORD.finditer(text)):
            word = fold(match.group())
            if word in self.stopwords:
                continue
            term = self.stemmer(word) if self.stemmer else word
            out.append(Token(term, position, match.start(), match.end()))
        return out

    def terms(self, text: str) -> List[str]:
        return [t.term for t in self.tokens(text)]
