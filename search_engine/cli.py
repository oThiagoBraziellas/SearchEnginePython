"""Command line interface: add, remove, search, stats."""

import argparse
import os
import sys
from typing import Iterator, List, Optional

from .index import Index

TEXT_EXTENSIONS = (".txt", ".md", ".rst")


def _files(paths: List[str]) -> Iterator[str]:
    for path in paths:
        if os.path.isdir(path):
            for root, dirs, names in os.walk(path):
                dirs.sort()
                for name in sorted(names):
                    if name.lower().endswith(TEXT_EXTENSIONS):
                        yield os.path.join(root, name)
        else:
            yield path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="search_engine", description="Full-text search with BM25")
    p.add_argument("--index", default="search.idx", help="index file (default: search.idx)")
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("add", help="index files or directories (.txt, .md, .rst)")
    a.add_argument("paths", nargs="+")
    r = sub.add_parser("remove", help="remove documents by name")
    r.add_argument("names", nargs="+")
    s = sub.add_parser("search", help='query: terms, +required, -excluded, "exact phrase"')
    s.add_argument("query")
    s.add_argument("-n", "--limit", type=int, default=10)
    sub.add_parser("stats", help="index statistics")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "search" and not os.path.exists(args.index):
        print(f"error: no index at {args.index!r}; run 'add' first", file=sys.stderr)
        return 1
    index = Index.load_or_create(args.index)
    if args.command == "add":
        count = 0
        for path in _files(args.paths):
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as handle:
                    index.add(path, handle.read())
            except OSError as exc:
                print(f"skipped {path}: {exc}", file=sys.stderr)
                continue
            count += 1
        index.save(args.index)
        print(f"indexed {count} document(s); total {len(index)}")
    elif args.command == "remove":
        removed = sum(1 for name in args.names if index.remove(name))
        index.save(args.index)
        print(f"removed {removed} document(s); total {len(index)}")
    elif args.command == "search":
        results = index.search(args.query, limit=args.limit)
        if not results:
            print("no results")
        for rank, hit in enumerate(results, 1):
            print(f"{rank:>2}. {hit.score:6.3f}  {hit.name}")
            print(f"            {hit.snippet}")
    else:
        for key, value in index.stats().items():
            print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
