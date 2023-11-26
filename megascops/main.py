import argparse
import json
import math
import os
from collections import defaultdict
from typing import Protocol
from PyPDF2 import PdfReader

RawTermFreq = dict[str, dict[str, int]]
TermDocMap = dict[str, set[str]]
TermFreqInverseDocFreq = dict[str, dict[str, float]]


class Tokenizer:
    def __init__(self, content: str) -> None:
        self._content = content

    def __str__(self) -> str:
        return self._content

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self._content[:100]}...')"

    def __iter__(self) -> "Tokenizer":
        return self

    def __next__(self) -> str:
        self._content = self._content.lstrip()
        if not len(self._content):
            raise StopIteration

        if self._content[0].isalnum():
            return self._chop_alphanum()

        return self._chop(1)

    def _chop(self, n: int) -> str:
        token = self._content[:n]
        self._content = self._content[n:]
        return token

    def _chop_alphanum(self) -> str:
        n = 0
        while n < len(self._content) and self._content[n].isalnum():
            n += 1
        token = self._chop(n)
        return token.lower()


class DocumentScanner(Protocol):
    def scan_directory(self) -> list[str]:
        ...

    @classmethod
    def extract_text(cls, filename: str) -> str:
        ...


class PdfDocumentScanner:
    def __init__(self, dirname: str) -> None:
        self._dirname = dirname

    def scan_directory(self) -> list[str]:
        filenames = list()
        root = os.fsencode(self._dirname)
        for file in os.listdir(root):
            filename = os.path.join(self._dirname, os.fsdecode(file))
            if filename.endswith(".pdf"):
                print("Got", filename)
                filenames.append(filename)
            else:
                print("Skipping", filename)
        return filenames

    @classmethod
    def extract_text(cls, filename: str) -> str:
        with open(filename, "rb") as f:
            reader = PdfReader(f)
            full_text_lines = []
            for page in reader.pages:
                full_text_lines.extend(page.extract_text().split("\n"))
            content = " ".join(full_text_lines)
        return content


class DocumentCorpus:
    def __init__(self, doc_scanner: DocumentScanner) -> None:
        self._doc_scanner = doc_scanner
        self._docnames = self._doc_scanner.scan_directory()
        self._len = len(self._docnames)

    def __str__(self) -> str:
        return "\n".join(self._docnames)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._dirname!r}/*.pdf)"

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> "DocumentCorpus":
        return self

    def __next__(self) -> tuple[str, str]:
        if not len(self._docnames):
            raise StopIteration
        docname = self._docnames.pop(0)
        doc_content = self._doc_scanner.extract_text(docname)
        return docname, doc_content


def index(dirname: str) -> None:
    rtf: RawTermFreq = dict()
    tdm: TermDocMap = defaultdict(set)
    corpus = DocumentCorpus(PdfDocumentScanner(dirname))
    for doc, text_content in corpus:
        rtf[doc] = defaultdict(int)
        for term in Tokenizer(text_content):
            rtf[doc][term] += 1
            tdm[term].add(doc)

    N = len(corpus)
    tfidf: TermFreqInverseDocFreq = defaultdict(dict)
    for doc, dtf in rtf.items():
        for term, raw_tf in dtf.items():
            tf = raw_tf / sum(dtf.values())
            idf = math.log(N / len(tdm[term]))
            tfidf[term][doc] = tf * idf

    with open("tfidf.json", "w") as f:
        json.dump(tfidf, f, ensure_ascii=False)


def search(query: str) -> None:
    try:
        with open("tfidf.json", "r") as f:
            tfidf: TermFreqInverseDocFreq = json.load(f)
    except FileNotFoundError:
        print("No index found")
        exit(1)

    cummulative_score = defaultdict(float)
    for term in Tokenizer(query):
        for doc, score in tfidf.get(term, {}).items():
            cummulative_score[doc] += score

    for doc, score in sorted(
        cummulative_score.items(), key=lambda x: x[1], reverse=True
    ):
        print(doc, score)


def main(args: argparse.Namespace) -> None:
    if args.command == "index":
        index(args.dirname)
        exit(0)
    if args.command == "search":
        search(args.query)
        exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="megascops", description="index and search PDF documents locally"
    )

    subparser = parser.add_subparsers(
        title="operations",
        description="subcommands",
        dest="command",
        help="subcommands",
    )
    index_subparser = subparser.add_parser(
        "index",
        help="index the documents in the directory",
    )
    index_subparser.add_argument("dirname", help="the directory to index")
    search_subparser = subparser.add_parser(
        "search",
        help="search the documents in the directory",
    )
    search_subparser.add_argument("query", help="the query to search for")

    args = parser.parse_args()

    main(args)
