"""Text processing: tokenization and heading-aware chunking.

Two production lessons are preserved deliberately:

* CJK text is tokenized into bigrams. Chinese has no spaces, so a whitespace
  tokenizer turns a whole sentence into one un-matchable term.
* The heading is folded into the chunk's tokens. Production tokenized the body
  only, which made a heading-only query unfindable through the lexical path.

One production bug is fixed: a note whose entire content is a heading produced
zero chunks, so it was invisible to retrieval *and* re-indexed on every run
(the self-heal heuristic could never converge for it).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TOKEN_PATTERN = re.compile(r"[\w-]+", re.UNICODE)
CJK_PATTERN = re.compile(r"[\u3400-\u9fff]+")
HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.+)")
FRONTMATTER_DELIMITER = "---"


@dataclass(frozen=True, slots=True)
class Chunk:
    heading: str
    content: str
    ordinal: int
    token_count: int
    body_chars: int


def tokenize(text: str) -> list[str]:
    terms = [term for term in TOKEN_PATTERN.findall(str(text).lower()) if len(term) > 1]
    tokens: set[str] = set()
    for term in terms:
        tokens.add(term)
        if CJK_PATTERN.fullmatch(term):
            tokens.update(term[index : index + 2] for index in range(len(term) - 1))
    return sorted(tokens)


def term_frequencies(tokens: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def split_frontmatter(content: str) -> tuple[str, str]:
    """Return (frontmatter_block, body). Both without the delimiters."""
    if not content.startswith(FRONTMATTER_DELIMITER):
        return "", content
    end = content.find(f"\n{FRONTMATTER_DELIMITER}", len(FRONTMATTER_DELIMITER))
    if end < 0:
        return "", content
    return content[len(FRONTMATTER_DELIMITER) : end], content[end + len(FRONTMATTER_DELIMITER) + 1 :]


def body_length(content: str) -> int:
    """Characters of real body text, ignoring frontmatter.

    This is the stub detector: production persona placeholders contain almost no
    body, yet their near-empty embeddings sat close to the corpus centroid and
    out-ranked substantive notes on unrelated queries.
    """
    _, body = split_frontmatter(content)
    return len("".join(body.split()))


def chunk_note(content: str, *, max_chars: int = 1800) -> list[Chunk]:
    _frontmatter, body = split_frontmatter(content)
    chunks: list[Chunk] = []
    heading = ""
    # The document title (the first H1) is the strongest single signal a note has,
    # and it lives *above* the first section. Treating only section headings as
    # context dropped it entirely: a note titled "从开题到成品的端到端流程" became
    # unfindable by its own title because no chunk carried those characters.
    document_title = ""
    lines: list[str] = []

    def flush() -> None:
        text = " ".join(lines).strip()
        if not text:
            return
        context = "\n".join(part for part in (document_title, heading) if part)
        tokens = tokenize(f"{context}\n{text}")
        chunks.append(
            Chunk(
                heading=heading,
                content=text,
                ordinal=len(chunks),
                token_count=len(tokens),
                body_chars=len("".join(text.split())),
            )
        )
        lines.clear()

    for line in body.splitlines():
        match = HEADING_PATTERN.match(line)
        if match:
            flush()
            level = len(line) - len(line.lstrip("#"))
            title = match.group(1).strip()
            if level == 1 and not document_title:
                document_title = title
            heading = title
        elif line.strip():
            lines.append(line.strip())
        if len(" ".join(lines)) > max_chars:
            flush()
    flush()

    if not chunks:
        # A note that is frontmatter and headings only would otherwise vanish and
        # be re-indexed forever. Fall back to its heading, then to its raw text.
        fallback = heading or " ".join(
            line.strip() for line in content.splitlines() if line.strip()
        )
        if fallback:
            tokens = tokenize(fallback)
            chunks.append(
                Chunk(
                    heading=heading,
                    content=fallback,
                    ordinal=0,
                    token_count=len(tokens),
                    body_chars=len("".join(fallback.split())),
                )
            )
    return chunks
