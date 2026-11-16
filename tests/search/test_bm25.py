"""Tests for the BM25 index over words and character n-grams.

The catalog below is made of real product names from the three chains.
"""

from math import log

import numpy as np
import pytest

from shopping_agent.search.bm25 import (
    BOUNDARY,
    Bm25Config,
    Bm25Index,
    Hit,
    ngrams,
    terms,
    words,
)

CATALOG = {
    "milk_3": "חלב תנובה 3% שומן 2 ליטר",
    "milk_1": "חלב טרי מהגולן 1% 1 ליטר",
    "milk_goat": "חלב עיזים 4% שומן 1 ל תנובה",
    "chocolate_milk": "שוקולד חלב מעולה",
    "chocolate_dark": "שוקולד מריר 70% 100 גרם",
    "cottage_5": "קוטג' 5% תנובה 250 ג' מהדרין",
    "cottage_9": "קוטג תנובה 9% שומן 250 ג",
    "bread": "לחם אחיד פרוס 750 גרם",
    "salt": 'מלח הארץ גס צנצנת 1 ק"ג',
    "olive_oil": 'שמן זית כתית מעולה 750 מ"ל',
    "canola_oil": "שמן קנולה 1 ליטר",
    "tuna": "טונה בשמן סטארקיסט 4*160 ג",
    "tomatoes": "עגבניות שרי אשכולות",
    "tomato_paste": "רסק עגבניות 22 בריקס 250",
    "yogurt": "יוגורט ביו 3% דנונה 200 ג",
    "cola": "coca cola zero 1.5 ליטר",
}


@pytest.fixture(scope="module")
def index() -> Bm25Index:
    return Bm25Index.build(list(CATALOG), CATALOG.values())


def found(index: Bm25Index, query: str, **options) -> list[str]:
    return [hit.key for hit in index.search(query, **options)]


class TestTerms:
    def test_words_are_split_and_lower_cased(self):
        assert words("Coca Cola ZERO 1.5 ליטר") == ["coca", "cola", "zero", "1", "5", "ליטר"]
        assert words("  חלב,תנובה/3%  ") == ["חלב", "תנובה", "3"]

    def test_hebrew_quote_marks_stay_inside_the_word(self):
        assert words("קוטג' 250 ג'") == ["קוטג'", "250", "ג'"]
        assert words('מלח 1 ק"ג') == ["מלח", "1", 'ק"ג']

    def test_trigrams_include_the_edges_of_the_word(self):
        assert ngrams("חלב") == [f"{BOUNDARY}חל", "חלב", f"לב{BOUNDARY}"]
        assert len(ngrams("שוקולד")) == 6

    def test_a_short_word_is_one_gram(self):
        assert ngrams("ל") == [f"{BOUNDARY}ל{BOUNDARY}"]
        assert ngrams("3") == [f"{BOUNDARY}3{BOUNDARY}"]

    def test_the_index_covers_both_words_and_ngrams(self, index):
        vocabulary = index.vocabulary.tolist()

        assert "w:חלב" in vocabulary
        assert "g:חלב" in vocabulary
        assert sum(term.startswith("w:") for term in vocabulary) > 50
        assert sum(term.startswith("g:") for term in vocabulary) > 150
        assert terms("חלב") == ["w:חלב", f"g:{BOUNDARY}חל", "g:חלב", f"g:לב{BOUNDARY}"]

    @pytest.mark.parametrize("text", [None, "", "   ", "!!! ---"])
    def test_text_without_words_has_no_terms(self, text):
        assert terms(text) == []


class TestSearch:
    def test_products_with_the_word_come_first(self, index):
        results = found(index, "חלב")

        assert set(results[:4]) == {"milk_3", "milk_1", "milk_goat", "chocolate_milk"}

    def test_more_matching_words_rank_higher(self, index):
        assert found(index, "חלב תנובה 3%")[0] == "milk_3"
        assert found(index, "שמן זית")[0] == "olive_oil"
        assert found(index, "רסק עגבניות")[0] == "tomato_paste"

    def test_a_rare_word_counts_more_than_a_common_one(self, index):
        # "תנובה" is on four products, "עיזים" on one.
        assert found(index, "תנובה עיזים")[0] == "milk_goat"

    def test_latin_text_is_matched_without_regard_to_case(self, index):
        assert found(index, "Coca Cola")[0] == "cola"

    def test_results_carry_the_key_the_position_and_the_score(self, index):
        hit = index.search("לחם")[0]

        assert hit == Hit(key="bread", position=list(CATALOG).index("bread"), score=hit.score)
        assert hit.score > 0
        assert type(hit.score) is float

    def test_results_are_ordered_by_score(self, index):
        scores = [hit.score for hit in index.search("חלב תנובה")]

        assert scores == sorted(scores, reverse=True)

    def test_products_sharing_nothing_with_the_query_are_not_returned(self, index):
        assert found(index, "xyzzy") == []
        assert "bread" not in found(index, "טונה")

    @pytest.mark.parametrize("query", [None, "", "   ", "%%%"])
    def test_an_empty_query_finds_nothing(self, index, query):
        assert index.search(query) == []

    def test_the_limit_cuts_the_results(self, index):
        assert len(index.search("חלב", limit=2)) == 2
        assert found(index, "חלב", limit=2) == found(index, "חלב")[:2]
        with pytest.raises(ValueError, match="at least 1"):
            index.search("חלב", limit=0)

    def test_equal_scores_keep_index_order(self):
        index = Bm25Index.build(["b", "a", "c"], ["חלב", "חלב", "חלב"])

        assert found(index, "חלב") == ["b", "a", "c"]

    def test_the_same_query_gives_the_same_results(self, index):
        assert index.search("חלב תנובה") == index.search("חלב תנובה")


class TestPrefixesAndSpellingVariants:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("וחלב", "milk"),
            ("בחלב", "milk"),
            ("לחלב", "milk"),
            ("והלחם", "bread"),
            ("מהמלח", "salt"),
            ("ושמן", "oil"),
        ],
    )
    def test_a_query_with_a_hebrew_prefix_still_retrieves_the_product(self, index, query, expected):
        assert f"w:{query}" not in index.vocabulary.tolist()

        assert expected in found(index, query)[0]

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("קוטג", "cottage"),
            ("קוטג'", "cottage"),
            ("קוטאג'", "cottage"),
            ("שוקלד", "chocolate"),
            ("יוגרט", "yogurt"),
            ("עגבניה", "tomato"),
            ("עגבניות", "tomato"),
            ("טונא", "tuna"),
        ],
    )
    def test_a_spelling_variant_still_retrieves_the_product(self, index, query, expected):
        assert expected in found(index, query)[0]

    def test_both_spellings_of_cottage_find_both_products(self, index):
        for query in ("קוטג", "קוטג'"):
            assert set(found(index, query)[:2]) == {"cottage_5", "cottage_9"}

    def test_an_exact_word_beats_a_word_that_only_shares_ngrams(self, index):
        # "מלח" is also inside no other product; "לחם" shares the letters "לח".
        assert found(index, "מלח")[0] == "salt"
        assert found(index, "לחם")[0] == "bread"

    def test_without_ngrams_a_prefixed_query_finds_nothing(self):
        words_only = Bm25Index.build(list(CATALOG), CATALOG.values(), Bm25Config(ngram_weight=0.0))

        assert found(words_only, "וחלב") == []
        assert found(words_only, "חלב")[0].startswith(("milk", "chocolate_milk"))


class TestMask:
    def test_the_mask_is_applied_before_the_cut(self, index):
        keys = list(CATALOG)
        only_goat = np.array([key == "milk_goat" for key in keys])
        best_without_mask = found(index, "חלב", limit=1)

        assert found(index, "חלב", limit=1, mask=only_goat) == ["milk_goat"]
        assert best_without_mask != ["milk_goat"] or len(found(index, "חלב")) > 1

    def test_masked_products_are_never_returned(self, index):
        mask = np.array([not key.startswith("milk") for key in CATALOG])

        results = found(index, "חלב", mask=mask)

        assert results == ["chocolate_milk"]

    def test_an_all_false_mask_finds_nothing(self, index):
        assert index.search("חלב", mask=np.zeros(len(index), dtype=np.bool_)) == []

    def test_a_mask_of_the_wrong_shape_or_type_is_rejected(self, index):
        with pytest.raises(ValueError, match="one boolean per indexed product"):
            index.search("חלב", mask=np.ones(3, dtype=np.bool_))
        with pytest.raises(ValueError, match="one boolean per indexed product"):
            index.search("חלב", mask=np.ones(len(index), dtype=np.int8))


class TestScoring:
    def test_the_score_of_a_single_word_follows_the_bm25_formula(self):
        index = Bm25Index.build(
            ["a", "b", "c"],
            ["חלב", "חלב טרי", "לחם"],
            Bm25Config(k1=1.2, b=0.75, word_weight=1.0, ngram_weight=0.0),
        )
        idf = log(1 + (3 - 2 + 0.5) / (2 + 0.5))
        average = (1 + 2 + 1) / 3

        scores = index.scores("חלב")

        def expected(length: int) -> float:
            return idf * 1 * 2.2 / (1 + 1.2 * (1 - 0.75 + 0.75 * length / average))

        assert scores.tolist() == pytest.approx([expected(1), expected(2), 0.0], rel=1e-6)
        assert scores.dtype == np.float32

    def test_a_shorter_name_scores_higher_for_the_same_word(self):
        index = Bm25Index.build(["short", "long"], ["חלב", "חלב תנובה 3% שומן 2 ליטר"])

        assert found(index, "חלב") == ["short", "long"]

    def test_a_repeated_query_word_counts_once(self, index):
        assert index.scores("חלב חלב חלב").tolist() == index.scores("חלב").tolist()

    @pytest.mark.parametrize(
        "fields",
        [
            {"k1": -1.0},
            {"b": 1.5},
            {"word_weight": -1.0},
            {"word_weight": 0.0, "ngram_weight": 0.0},
        ],
    )
    def test_bad_parameters_are_rejected(self, fields):
        with pytest.raises(ValueError):
            Bm25Config(**fields)


class TestBuild:
    def test_one_entry_per_product(self, index):
        assert len(index) == len(CATALOG)
        assert index.keys.tolist() == list(CATALOG)

    def test_a_product_without_a_name_is_indexed_but_never_found(self):
        index = Bm25Index.build(["a", "b"], [None, "חלב"])

        assert len(index) == 2
        assert found(index, "חלב") == ["b"]

    def test_an_empty_catalog_can_be_built_and_searched(self):
        index = Bm25Index.build([], [])

        assert len(index) == 0
        assert index.search("חלב") == []

    def test_keys_must_be_unique_and_match_the_names(self):
        with pytest.raises(ValueError, match="must be unique"):
            Bm25Index.build(["a", "a"], ["חלב", "לחם"])
        with pytest.raises(ValueError, match="2 keys were given for 1 names"):
            Bm25Index.build(["a", "b"], ["חלב"])

    def test_names_can_come_from_a_generator(self):
        index = Bm25Index.build(["a", "b"], (name for name in ("חלב", "לחם")))

        assert found(index, "לחם") == ["b"]


class TestSaveAndLoad:
    def test_the_index_is_saved_as_one_file(self, index, tmp_path):
        path = index.save(tmp_path / "bm25.npz")

        assert path == tmp_path / "bm25.npz"
        assert [file.name for file in tmp_path.iterdir()] == ["bm25.npz"]
        assert path.stat().st_size > 0

    def test_a_loaded_index_answers_like_the_one_that_was_saved(self, index, tmp_path):
        loaded = Bm25Index.load(index.save(tmp_path / "bm25.npz"))

        assert len(loaded) == len(index)
        assert loaded.config == index.config
        for query in ("חלב", "וחלב", "קוטג", "שמן זית", "coca cola", "xyzzy"):
            assert loaded.search(query) == index.search(query)

    def test_the_parameters_are_saved_with_the_index(self, tmp_path):
        config = Bm25Config(k1=0.9, b=0.4, word_weight=2.0, ngram_weight=0.5)
        index = Bm25Index.build(list(CATALOG), CATALOG.values(), config)

        loaded = Bm25Index.load(index.save(tmp_path / "bm25.npz"))

        assert loaded.config == config
        assert loaded.scores("חלב").tolist() == index.scores("חלב").tolist()

    def test_the_file_holds_plain_arrays_only(self, index, tmp_path):
        path = index.save(tmp_path / "bm25.npz")

        with np.load(path, allow_pickle=False) as data:
            assert set(data.files) == {
                "format_version",
                "keys",
                "vocabulary",
                "offsets",
                "documents",
                "weights",
                "config",
            }

    def test_the_file_name_must_end_with_npz(self, index, tmp_path):
        with pytest.raises(ValueError, match="must end with .npz"):
            index.save(tmp_path / "bm25.bin")

    def test_a_file_of_another_format_version_is_rejected(self, index, tmp_path):
        path = index.save(tmp_path / "bm25.npz")
        with np.load(path, allow_pickle=False) as data:
            arrays = dict(data)
        arrays["format_version"] = np.array(99)
        np.savez_compressed(path, **arrays)

        with pytest.raises(ValueError, match="index format 99 is not supported"):
            Bm25Index.load(path)

    def test_an_empty_index_round_trips(self, tmp_path):
        loaded = Bm25Index.load(Bm25Index.build([], []).save(tmp_path / "empty.npz"))

        assert len(loaded) == 0
        assert loaded.search("חלב") == []
