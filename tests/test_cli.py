import contextlib
import io
import os
import tempfile
import unittest

from search_engine.cli import main

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples", "docs")


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = main(list(argv))
    return rc, out.getvalue(), err.getvalue()


class CliTests(unittest.TestCase):
    def test_add_search_remove_with_bundled_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            idx = os.path.join(tmp, "s.idx")
            rc, out, _ = run("--index", idx, "add", SAMPLES)
            self.assertEqual(rc, 0)
            self.assertIn("indexed 6 document(s)", out)

            rc, out, _ = run("--index", idx, "search", '"consistent hashing"')
            self.assertEqual(rc, 0)
            names = [line.split()[-1] for line in out.splitlines() if line[:3].strip().rstrip(".").isdigit()]
            self.assertTrue(names[0].endswith("consistent-hashing.txt"), out)
            self.assertIn("replication.txt", "".join(names))

            rc, out, _ = run("--index", idx, "search", "+cache -eviction")
            self.assertNotIn("caching.txt", out)

            self.assertIn("no results", run("--index", idx, "search", "zebra")[1])

            target = os.path.join(SAMPLES, "caching.txt")
            self.assertIn("removed 1", run("--index", idx, "remove", target)[1])
            self.assertIn("documents: 5", run("--index", idx, "stats")[1])

    def test_search_without_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(run("--index", os.path.join(tmp, "none"), "search", "x")[0], 1)

    def test_missing_file_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, out, err = run("--index", os.path.join(tmp, "s.idx"), "add", "/no/such/file.txt")
            self.assertEqual(rc, 0)
            self.assertIn("skipped", err)
            self.assertIn("indexed 0", out)


if __name__ == "__main__":
    unittest.main()
