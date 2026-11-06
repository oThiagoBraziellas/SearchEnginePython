import math
import os
import tempfile
import unittest

from search_engine import Index, Tokenizer, parse_query

DOCS = {
    "lru": "The least recently used cache evicts the entry that was read the longest time ago.",
    "ring": "Consistent hashing places servers on a ring. Virtual nodes even out the load of each server.",
    "wal": "A write ahead log is replayed after a crash. Compaction rewrites the log.",
    "cache2": "Cache invalidation is hard. A cache without invalidation serves stale data. Cache cache cache.",
    "recent": "Recently used items are kept; the least popular ones are used last.",
}


def build(**kw):
    index = Index(**kw)
    for name, text in DOCS.items():
        index.add(name, text)
    return index


def names(results):
    return [r.name for r in results]


class QueryParsingTests(unittest.TestCase):
    def setUp(self):
        self.tok = Tokenizer(stemmer=None)

    def test_clauses(self):
        q = parse_query('cache +eviction -redis "least recently used" -"write log"', self.tok)
        self.assertEqual(q.optional, ["cache"])
        self.assertEqual(q.required, ["eviction"])
        self.assertEqual(q.excluded, ["redis"])
        self.assertEqual(q.phrases, [[("least", 0), ("recently", 1), ("used", 2)]])
        self.assertEqual(q.excluded_phrases, [[("write", 0), ("log", 1)]])

    def test_stopwords_inside_phrase_keep_their_gap(self):
        q = parse_query('"cache of the entry"', self.tok)
        self.assertEqual(q.phrases, [[("cache", 0), ("entry", 3)]])

    def test_empty_and_stopword_only_queries(self):
        for text in ["", "   ", "the of and", '""', "+ -"]:
            self.assertFalse(parse_query(text, self.tok).has_positive_clause, text)

    def test_quoted_single_term_is_required(self):
        self.assertEqual(parse_query('"cache"', self.tok).required, ["cache"])


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.index = build()

    def test_single_term(self):
        self.assertEqual(set(names(self.index.search("ring"))), {"ring"})

    def test_no_match_and_empty_query(self):
        self.assertEqual(self.index.search("zebra"), [])
        self.assertEqual(self.index.search(""), [])
        self.assertEqual(self.index.search("the of"), [])
        self.assertEqual(Index().search("anything"), [])

    def test_stemming_matches_variants(self):
        self.assertIn("ring", names(self.index.search("servers")))
        self.assertIn("ring", names(self.index.search("server")))
        self.assertEqual(names(self.index.search("evicting")), ["lru"])  # doc says "evicts"
        self.assertEqual(names(self.index.search("replaying")), ["wal"])  # doc says "replayed"
        self.assertEqual(names(self.index.search("replayed")), ["wal"])

    def test_case_and_accent_insensitive(self):
        index = Index()
        index.add("pt", "Configuração do serviço de autenticação")
        self.assertEqual(names(index.search("CONFIGURACAO")), ["pt"])
        self.assertEqual(names(index.search("autenticação")), ["pt"])

    def test_or_semantics_rank_documents_with_more_terms_higher(self):
        results = self.index.search("least recently used cache")
        self.assertEqual(results[0].name, "lru")
        self.assertGreater(len(results), 1)
        self.assertTrue(all(a.score >= b.score for a, b in zip(results, results[1:])))

    def test_required_terms(self):
        self.assertEqual(names(self.index.search("+cache +entry")), ["lru"])
        self.assertEqual(self.index.search("+cache +zebra"), [])

    def test_excluded_terms(self):
        with_all = set(names(self.index.search("cache least")))
        without = set(names(self.index.search("cache least -invalidation")))
        self.assertIn("cache2", with_all)
        self.assertEqual(without, with_all - {"cache2"})

    def test_only_exclusion_returns_nothing(self):
        self.assertEqual(self.index.search("-cache"), [])

    def test_phrase_requires_adjacency(self):
        # "least recently used" appears as a phrase only in "lru"; "recent" has the words apart.
        self.assertEqual(names(self.index.search('"least recently used"')), ["lru"])
        self.assertEqual(names(self.index.search('"recently used least"')), [])
        self.assertEqual(set(names(self.index.search("least recently used"))) >= {"lru", "recent"}, True)

    def test_phrase_does_not_match_across_a_removed_stopword(self):
        index = Index()
        index.add("a", "write ahead log")
        index.add("b", "write the ahead log")
        self.assertEqual(names(index.search('"write ahead"')), ["a"])
        self.assertEqual(names(index.search('"write the ahead"')), ["b"])  # gap is preserved

    def test_phrase_with_repeated_terms(self):
        index = Index(stopwords=None)
        index.add("a", "buffalo buffalo buffalo")
        index.add("b", "buffalo x buffalo")
        self.assertEqual(names(index.search('"buffalo buffalo"')), ["a"])

    def test_excluded_phrase(self):
        index = build()
        results = names(index.search('cache -"least recently used"'))
        self.assertNotIn("lru", results)
        self.assertIn("cache2", results)

    def test_limit_and_deterministic_ties(self):
        index = Index()
        for name in ["c", "a", "b"]:
            index.add(name, "same text")
        self.assertEqual(names(index.search("same")), ["a", "b", "c"])
        self.assertEqual(names(index.search("same", limit=2)), ["a", "b"])

    def test_snippet_contains_match(self):
        hit = self.index.search("compaction")[0]
        self.assertIn("Compaction", hit.snippet)
        long = Index()
        long.add("x", "filler " * 40 + "needle " + "filler " * 40)
        snippet = long.search("needle", snippet_width=40)[0].snippet
        self.assertTrue(snippet.startswith("...") and snippet.endswith("..."))
        self.assertIn("needle", snippet)


class BM25Tests(unittest.TestCase):
    def test_score_matches_formula(self):
        index = Index(stopwords=None, stemming=False, k1=1.2, b=0.75)
        index.add("d1", "apple banana apple")
        index.add("d2", "banana cherry")
        index.add("d3", "cherry cherry cherry date")
        n, df, tf = 3, 1, 2
        avg = (3 + 2 + 4) / 3
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        expected = idf * tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * 3 / avg))
        result = index.search("apple")[0]
        self.assertEqual(result.name, "d1")
        self.assertAlmostEqual(result.score, expected)

    def test_rare_terms_weigh_more(self):
        index = Index(stopwords=None, stemming=False)
        index.add("d1", "common rare")
        index.add("d2", "common other")
        index.add("d3", "common thing")
        self.assertEqual(names(index.search("common rare"))[0], "d1")
        self.assertGreater(index._idf("rare"), index._idf("common"))

    def test_term_frequency_saturates(self):
        index = Index(stopwords=None, stemming=False)
        index.add("few", "x " + "pad " * 8)
        index.add("many", "x " * 4 + "pad " * 5)
        index.add("lots", "x " * 9)
        index.add("other", "pad " * 9)
        scores = {r.name: r.score for r in index.search("x")}
        gain_1 = scores["many"] - scores["few"]
        gain_2 = scores["lots"] - scores["many"]
        self.assertGreater(gain_1, 0)
        self.assertLess(gain_2, gain_1 * 1.5)  # diminishing returns, not linear

    def test_shorter_documents_win_with_equal_tf(self):
        index = Index(stopwords=None, stemming=False)
        index.add("short", "target word")
        index.add("long", "target " + "filler " * 30)
        index.add("none", "nothing here")
        self.assertEqual(names(index.search("target"))[0], "short")

    def test_idf_never_negative(self):
        index = Index(stopwords=None, stemming=False)
        for i in range(5):
            index.add(f"d{i}", "everywhere")
        self.assertGreater(index.search("everywhere")[0].score, 0)


class MutationTests(unittest.TestCase):
    def test_replace_and_remove(self):
        index = build()
        index.add("ring", "Completely different content about bananas")
        self.assertEqual(len(index), len(DOCS))
        self.assertEqual(index.search("virtual"), [])
        self.assertEqual(names(index.search("bananas")), ["ring"])
        self.assertTrue(index.remove("ring"))
        self.assertFalse(index.remove("ring"))
        self.assertEqual(index.search("bananas"), [])
        self.assertNotIn("ring", index)

    def test_remove_cleans_vocabulary_and_stats(self):
        index = Index()
        index.add("only", "uniqueword")
        index.remove("only")
        self.assertEqual(index.vocabulary_size, 0)
        self.assertEqual(index.stats()["documents"], 0)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            Index(b=1.5)
        with self.assertRaises(ValueError):
            Index(k1=-1)


class PersistenceTests(unittest.TestCase):
    def test_round_trip_gives_identical_results(self):
        index = build()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "idx.gz")
            index.save(path)
            self.assertFalse(os.path.exists(path + ".tmp"))
            loaded = Index.load(path)
        for q in ["cache", '"least recently used"', "+cache -invalidation", "server ring load", "zebra"]:
            a, b = index.search(q), loaded.search(q)
            self.assertEqual([(r.name, round(r.score, 9), r.snippet) for r in a],
                             [(r.name, round(r.score, 9), r.snippet) for r in b], q)
        self.assertEqual(index.stats(), loaded.stats())

    def test_loaded_index_can_be_updated(self):
        index = build()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "idx.gz")
            index.save(path)
            loaded = Index.load(path)
            loaded.add("new", "brand new document about sharding")
            loaded.remove("wal")
            loaded.save(path)
            again = Index.load(path)
        self.assertEqual(names(again.search("sharding")), ["new"])
        self.assertEqual(again.search("crash"), [])

    def test_settings_survive(self):
        index = Index(k1=2.0, b=0.3, stemming=False, stopwords=["foo"])
        index.add("d", "foo bar running")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "idx.gz")
            index.save(path)
            loaded = Index.load(path)
        self.assertEqual((loaded.k1, loaded.b), (2.0, 0.3))
        self.assertEqual(loaded.search("foo"), [])
        self.assertEqual(names(loaded.search("running")), ["d"])
        self.assertEqual(loaded.search("run"), [])  # stemming stayed off

    def test_rejects_unknown_format(self):
        import gzip
        import json

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "idx.gz")
            with gzip.open(path, "wt") as f:
                json.dump({"version": 99}, f)
            with self.assertRaises(ValueError):
                Index.load(path)

    def test_load_or_create(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(len(Index.load_or_create(os.path.join(tmp, "missing"))), 0)


class ScaleTests(unittest.TestCase):
    def test_thousand_documents(self):
        index = Index()
        for i in range(1000):
            index.add(f"doc{i}", f"common words here topic{i % 50} unique{i}")
        self.assertEqual(names(index.search("unique777")), ["doc777"])
        self.assertEqual(len(index.search("topic7", limit=100)), 20)
        self.assertEqual(len(index.search('"common words here"', limit=5)), 5)


if __name__ == "__main__":
    unittest.main()
