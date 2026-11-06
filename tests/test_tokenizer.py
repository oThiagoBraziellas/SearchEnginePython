import unittest

from search_engine import Tokenizer, simple_stem
from search_engine.tokenizer import fold


class TokenizerTests(unittest.TestCase):
    def test_lowercase_and_punctuation(self):
        t = Tokenizer(stopwords=None, stemmer=None)
        self.assertEqual(t.terms("Hello, World! It's 2026."), ["hello", "world", "it", "s", "2026"])

    def test_accents_are_folded(self):
        self.assertEqual(fold("Configuração Ação São"), "configuracao acao sao")
        t = Tokenizer(stopwords=None, stemmer=None)
        self.assertEqual(t.terms("Ação"), t.terms("acao"))

    def test_stopwords_removed_but_positions_kept(self):
        t = Tokenizer(stemmer=None)
        tokens = t.tokens("the cat of the house")
        self.assertEqual([(x.term, x.position) for x in tokens], [("cat", 1), ("house", 4)])

    def test_offsets_point_into_the_original_text(self):
        text = "Ação rápida"
        for tok in Tokenizer(stopwords=None, stemmer=None).tokens(text):
            self.assertEqual(fold(text[tok.start : tok.end]), tok.term)

    def test_stemmer_groups_variants(self):
        groups = [
            ["cache", "caches", "cached", "caching"],
            ["queue", "queues"],
            ["query", "queries"],
            ["replicate", "replicated"],
            ["run", "runs", "running"],
        ]
        for group in groups:
            self.assertEqual(len({simple_stem(w) for w in group}), 1, group)

    def test_stemmer_keeps_short_and_special_words(self):
        for word in ["is", "bus", "class", "was", "go", "gas", "this", "status"]:
            self.assertEqual(simple_stem(word), word)

    def test_empty_text(self):
        self.assertEqual(Tokenizer().tokens(""), [])
        self.assertEqual(Tokenizer().tokens("!!! ... ---"), [])


if __name__ == "__main__":
    unittest.main()
