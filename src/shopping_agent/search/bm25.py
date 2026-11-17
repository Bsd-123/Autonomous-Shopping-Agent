"""The lexical index: BM25 over word tokens and character n-grams (architecture section 2).

Every product name is indexed twice:

* by its **words**, which carry most of the weight, and
* by the **character trigrams** of each word, which let a query still reach a
  product when it is written with a Hebrew prefix ("וחלב" for "חלב") or with a
  spelling variant ("קוטג" for "קוטג'", "שוקלד" for "שוקולד").

The index is built offline from ``name_normalized`` and saved as one ``.npz``
file for the image build. It needs no service: a query adds up a few NumPy
slices.

Each posting already holds the full BM25 weight of its term in its product,
so answering a query is a sum over the query's terms.
"""

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import log
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray

NGRAM_SIZE = 3
# The two kinds of term are kept apart in the vocabulary by a prefix.
WORD = "w:"
NGRAM = "g:"
# A word is padded with this mark on both sides, so that its first and last
# trigram differ from the same letters in the middle of another word.
BOUNDARY = "\x1f"
FORMAT_VERSION = 1

# Letters and digits, with the Hebrew geresh and gershayim kept inside a word.
_WORD = re.compile(r"[^\W_]+(?:['\"][^\W_]+)*'?")


@dataclass(frozen=True, slots=True)
class Bm25Config:
    """The parameters of the index.

    ``k1`` and ``b`` are the usual BM25 parameters. A query term that is a
    whole word counts ``word_weight``; a trigram counts ``ngram_weight``, which
    is lower because one word shares several trigrams.
    """

    k1: float = 1.2
    b: float = 0.75
    word_weight: float = 1.0
    ngram_weight: float = 0.3

    def __post_init__(self) -> None:
        if self.k1 < 0 or not 0 <= self.b <= 1:
            raise ValueError("BM25 needs k1 >= 0 and 0 <= b <= 1")
        if self.word_weight < 0 or self.ngram_weight < 0:
            raise ValueError("term weights cannot be negative")
        if self.word_weight == 0 and self.ngram_weight == 0:
            raise ValueError("at least one of the term weights must be positive")


def words(text: str | None) -> list[str]:
    """The words of a name, lower-cased, in order."""
    return _WORD.findall(text.casefold()) if text else []


def ngrams(word: str) -> list[str]:
    """The character trigrams of one word, including its two edges."""
    padded = f"{BOUNDARY}{word}{BOUNDARY}"
    if len(padded) <= NGRAM_SIZE:
        return [padded]
    return [padded[start : start + NGRAM_SIZE] for start in range(len(padded) - NGRAM_SIZE + 1)]


def terms(text: str | None) -> list[str]:
    """Every index term of a name: its words, and the trigrams of each word."""
    found: list[str] = []
    for word in words(text):
        found.append(WORD + word)
        found.extend(NGRAM + gram for gram in ngrams(word))
    return found


@dataclass(frozen=True, slots=True)
class Hit:
    """One search result: the product's key, its position in the index and its score."""

    key: str
    position: int
    score: float


class Bm25Index:
    """A BM25 index over product names, built once and then only read."""

    def __init__(
        self,
        keys: NDArray[np.str_],
        vocabulary: NDArray[np.str_],
        offsets: NDArray[np.int64],
        documents: NDArray[np.int32],
        weights: NDArray[np.float32],
        config: Bm25Config,
    ) -> None:
        self.keys = keys
        self.vocabulary = vocabulary
        self.offsets = offsets
        self.documents = documents
        self.weights = weights
        self.config = config
        self._term_ids = {term: number for number, term in enumerate(vocabulary.tolist())}

    def __len__(self) -> int:
        return len(self.keys)

    @classmethod
    def build(
        cls, keys: Sequence[str], names: Iterable[str | None], config: Bm25Config | None = None
    ) -> Self:
        """Index one name per key. ``names`` are the normalized product names."""
        config = config or Bm25Config()
        keys = list(keys)
        if len(set(keys)) != len(keys):
            raise ValueError("the keys of the index must be unique")
        counts = [Counter(terms(name)) for name in names]
        if len(counts) != len(keys):
            raise ValueError(f"{len(keys)} keys were given for {len(counts)} names")

        # A product's length is counted separately for words and for trigrams,
        # so that a long name is not penalized twice.
        lengths = {
            kind: np.array(
                [sum(n for term, n in count.items() if term.startswith(kind)) for count in counts],
                dtype=np.float64,
            )
            for kind in (WORD, NGRAM)
        }
        average = {
            kind: float(values.mean()) if len(values) else 0.0 for kind, values in lengths.items()
        }

        postings: dict[str, list[tuple[int, int]]] = {}
        for document, count in enumerate(counts):
            for term, frequency in count.items():
                postings.setdefault(term, []).append((document, frequency))

        vocabulary = sorted(postings)
        offsets = np.zeros(len(vocabulary) + 1, dtype=np.int64)
        documents: list[int] = []
        weights: list[float] = []
        total = len(keys)
        for number, term in enumerate(vocabulary):
            kind = WORD if term.startswith(WORD) else NGRAM
            entries = postings[term]
            idf = log(1 + (total - len(entries) + 0.5) / (len(entries) + 0.5))
            for document, frequency in entries:
                norm = 1 - config.b
                if average[kind] > 0:
                    norm += config.b * lengths[kind][document] / average[kind]
                documents.append(document)
                weights.append(idf * frequency * (config.k1 + 1) / (frequency + config.k1 * norm))
            offsets[number + 1] = len(documents)
        return cls(
            keys=np.array(keys, dtype=np.str_),
            vocabulary=np.array(vocabulary, dtype=np.str_),
            offsets=offsets,
            documents=np.array(documents, dtype=np.int32),
            weights=np.array(weights, dtype=np.float32),
            config=config,
        )

    def scores(self, query: str | None) -> NDArray[np.float32]:
        """The BM25 score of every product for the query, in index order."""
        totals = np.zeros(len(self), dtype=np.float32)
        for term in dict.fromkeys(terms(query)):
            number = self._term_ids.get(term)
            if number is None:
                continue
            weight = self.config.word_weight if term.startswith(WORD) else self.config.ngram_weight
            if weight == 0:
                continue
            start, end = self.offsets[number], self.offsets[number + 1]
            # A product appears at most once in a term's postings.
            totals[self.documents[start:end]] += weight * self.weights[start:end]
        return totals

    def search(
        self, query: str | None, *, limit: int = 200, mask: NDArray[np.bool_] | None = None
    ) -> list[Hit]:
        """The best-scoring products for the query, best first.

        ``mask`` marks the products that may be returned. It is applied to the
        whole catalog before the cut to ``limit``, so a hard filter never loses
        a product to a better-scoring one that it excludes. Products that share
        no term with the query are not returned. Equal scores keep index order.
        """
        if limit < 1:
            raise ValueError("limit must be at least 1")
        totals = self.scores(query)
        if mask is not None:
            if mask.shape != totals.shape or mask.dtype != np.bool_:
                raise ValueError("the mask must hold one boolean per indexed product")
            totals = np.where(mask, totals, 0)
        matched = np.flatnonzero(totals > 0)
        order = matched[np.argsort(-totals[matched], kind="stable")][:limit]
        return [Hit(str(self.keys[i]), int(i), float(totals[i])) for i in order]

    def save(self, path: Path) -> Path:
        """Write the index to one ``.npz`` file and return its path."""
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError("the index file must end with .npz")
        config = self.config
        with path.open("wb") as file:
            np.savez_compressed(
                file,
                format_version=np.array(FORMAT_VERSION),
                keys=self.keys,
                vocabulary=self.vocabulary,
                offsets=self.offsets,
                documents=self.documents,
                weights=self.weights,
                config=np.array([config.k1, config.b, config.word_weight, config.ngram_weight]),
            )
        return path

    @classmethod
    def load(cls, path: Path) -> Self:
        """Read an index written by ``save``."""
        with np.load(Path(path), allow_pickle=False) as data:
            version = int(data["format_version"])
            if version != FORMAT_VERSION:
                raise ValueError(f"index format {version} is not supported; rebuild the index")
            k1, b, word_weight, ngram_weight = (float(value) for value in data["config"])
            return cls(
                keys=data["keys"],
                vocabulary=data["vocabulary"],
                offsets=data["offsets"],
                documents=data["documents"],
                weights=data["weights"],
                config=Bm25Config(k1, b, word_weight, ngram_weight),
            )
