"""Index package: schema, text processing, embeddings, indexing, retrieval."""

from .embed import Embedder, EmbedderUnavailable, FastEmbedEmbedder, LexicalOnlyEmbedder, ModelInfo
from .indexer import Indexer, IndexReport, index_vault
from .schema import SCHEMA_VERSION, initialize, status
from .searcher import Hit, Query, SearchDiagnostics, Searcher, search_index
from .text import Chunk, body_length, chunk_note, split_frontmatter, term_frequencies, tokenize

__all__ = [
    "SCHEMA_VERSION",
    "Chunk",
    "Embedder",
    "EmbedderUnavailable",
    "FastEmbedEmbedder",
    "Hit",
    "IndexReport",
    "Indexer",
    "LexicalOnlyEmbedder",
    "ModelInfo",
    "Query",
    "SearchDiagnostics",
    "Searcher",
    "body_length",
    "chunk_note",
    "index_vault",
    "initialize",
    "search_index",
    "split_frontmatter",
    "status",
    "term_frequencies",
    "tokenize",
]
