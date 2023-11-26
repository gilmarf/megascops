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
    def __init__(self, content: str):
        self._content = content

    def __str__(self):
        return self._content

    def __repr__(self):
        return f"{self.__class__.__name__}('{self._content[:100]}...')"

    def __iter__(self):
        return self

    def __next__(self):
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


def main():
    rtf: RawTermFreq = dict()
    tdm: TermDocMap = defaultdict(set)
    corpus = DocumentCorpus(PdfDocumentScanner("../misc"))
    for doc, text_content in corpus:
        rtf[doc] = defaultdict(int)
        for term in Tokenizer(text_content):
            rtf[doc][term] += 1
            tdm[term].add(doc)

    N = len(corpus)
    tfidf: TermFreqInverseDocFreq = dict()
    for doc, dtf in rtf.items():
        tfidf[doc] = dict()
        for term, raw_tf in dtf.items():
            tf = raw_tf / sum(dtf.values())
            idf = math.log(N / len(tdm[term]))
            tfidf[doc][term] = tf * idf

    with open("tfidf.json", "w") as f:
        json.dump(tfidf, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
