"""MCP stdio adapter.

This is the only module that knows about the wire. Its job is to render
:class:`GatewayError` faithfully: the SDK's default behaviour is to replace any
plain exception with ``"Error executing tool X"``, which discards the message and
the recovery instruction. A caller that is told only "error" cannot re-read the
note and retry, so the optimistic-concurrency contract silently stops working.

Errors are therefore returned as a successful tool call whose payload carries
``isError`` semantics in a structured body, and only genuinely unexpected
exceptions are allowed to escape to the SDK.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from ..core.audit import AuditLog
from ..core.config import Settings, load_settings
from ..core.errors import GatewayError
from ..index.embed import LexicalOnlyEmbedder, build_embedder
from ..index.indexer import Indexer
from ..index.searcher import Searcher
from ..services.archive import ArchiveService
from ..services.notes import NoteService
from ..services.review import ReviewService
from ..tools.registry import ToolSpec, build_tools


def build_registry(settings: Settings | None = None) -> tuple[dict[str, ToolSpec], Settings]:
    settings = settings or load_settings()
    settings.state_root.mkdir(parents=True, exist_ok=True)
    audit = AuditLog(settings.audit_log, agent_label=settings.agent_label)
    try:
        embedder = build_embedder(settings.embed_model, enabled=settings.semantic_enabled)
    except Exception:  # noqa: BLE001 - a missing model must degrade, not crash startup
        embedder = LexicalOnlyEmbedder()

    notes = NoteService(
        root=settings.vault_root,
        inbox_folders=settings.inbox_folders,
        lock_dir=settings.lock_dir,
        audit=audit,
        stale_lock_seconds=settings.stale_lock_seconds,
    )
    archive = ArchiveService(
        root=settings.vault_root,
        audit=audit,
        archive_dirname=settings.archive_dirname,
    )
    review = ReviewService(root=settings.vault_root)
    indexer = Indexer(settings.vault_root, settings.rag_db, embedder=embedder)
    searcher = Searcher(
        settings.rag_db,
        embedder=embedder,
        rrf_k=settings.rrf_k,
        rrf_depth=settings.rrf_depth,
        semantic_floor=settings.semantic_floor,
        semantic_min_tokens=settings.semantic_min_tokens,
        stub_body_chars=settings.stub_body_chars,
        stub_penalty=settings.stub_penalty,
    )
    return (
        build_tools(
            notes=notes,
            archive=archive,
            review=review,
            indexer=indexer,
            searcher=searcher,
            embedder=embedder,
            lock_dir=settings.lock_dir,
        ),
        settings,
    )


def _error_payload(error: GatewayError) -> dict:
    return {"isError": True, **error.to_payload()}


def main() -> None:  # pragma: no cover - exercised by e2e tests in a subprocess
    from mcp.server.mcpserver import MCPServer

    tools, settings = build_registry()
    server = MCPServer("secondbrain-mcp")

    for name, spec in tools.items():
        _register(server, name, spec)

    _ = settings
    server.run()


def _register(server, name: str, spec: ToolSpec) -> dict:
    """Register one tool with a signature derived from its declared schema.

    The SDK infers the JSON schema from the function signature, so a bare
    ``**kwargs`` wrapper would publish a tool with no arguments and every client
    would reject the real ones. The wrapper dispatches through ``**kwargs`` but
    carries a synthesized ``__signature__`` that the SDK reads.
    """
    import inspect

    properties = spec.input_schema.get("properties", {})
    required = set(spec.input_schema.get("required", []))
    parameters = [
        inspect.Parameter(
            argument,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=inspect.Parameter.empty if argument in required else None,
            annotation=_annotation_for(schema),
        )
        for argument, schema in properties.items()
    ]

    def handler(**kwargs: Any) -> dict:
        try:
            # The SDK passes every declared property, including omitted optional
            # ones as None. Dropping them lets each service's own defaults apply
            # instead of being overwritten by an explicit None.
            return spec.handler(**{key: value for key, value in kwargs.items() if value is not None})
        except GatewayError as error:
            return _error_payload(error)

    handler.__name__ = name
    handler.__doc__ = spec.description
    handler.__signature__ = inspect.Signature(parameters)  # type: ignore[attr-defined]
    server.tool(name=name, description=spec.description)(handler)
    return {"name": name, "description": spec.description, "inputSchema": spec.input_schema}


def _annotation_for(schema: dict):
    return {
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }.get(schema.get("type"), str)


def dispatch(name: str, arguments: dict, tools: dict[str, ToolSpec]) -> dict:
    """In-process dispatch used by tests and the CLI.

    Kept separate from the MCP wiring so the same contract can be asserted
    without a subprocess.
    """
    spec = tools.get(name)
    if spec is None:
        return {
            "isError": True,
            "error": {
                "code": "UNKNOWN_TOOL",
                "message": f"Unknown tool: {name}",
                "recovery": f"Available tools: {sorted(tools)}",
            },
        }
    try:
        return spec.handler(**arguments)
    except GatewayError as error:
        return _error_payload(error)
    except TypeError as error:
        return {
            "isError": True,
            "error": {
                "code": "INVALID_ARGUMENTS",
                "message": str(error),
                "recovery": "Check the tool schema for required fields",
            },
        }


def tool_manifest(tools: dict[str, ToolSpec]) -> list[dict]:
    return [
        {"name": spec.name, "description": spec.description, "inputSchema": spec.input_schema}
        for spec in tools.values()
    ]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


_ = json
