"""End-to-end test: real MCP stdio server subprocess + official client session."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO_ROOT = Path(__file__).resolve().parents[1]


def _payload(result):
    return json.loads(result.content[0].text)


def test_stdio_server_end_to_end(tmp_path):
    env = {
        **os.environ,
        "SB_VAULT_ROOT": str(REPO_ROOT / "demo-vault"),
        "SB_RAG_DB": str(tmp_path / "demo.db"),
        "SB_AGENT_LABEL": "e2e",
    }
    params = StdioServerParameters(command=sys.executable, args=["-m", "secondbrain_mcp.server"], env=env)

    async def run():
        async with (
            stdio_client(params) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()

            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            assert {
                "read_note",
                "list_notes",
                "search_notes",
                "rag_search",
                "rag_reindex",
                "rag_status",
                "create_inbox_note",
                "append_note",
                "update_metadata",
                "move_note",
                "archive_notes",
                "restore_archive",
            } <= names

            reindex = _payload(await session.call_tool("rag_reindex", {}))
            assert reindex["indexed"] >= 9

            search = _payload(await session.call_tool("rag_search", {"query": "望远镜"}))
            assert search["hits"][0]["source"] == "10_Notes/设备清单.md"
            assert search["requires_read_note_for_full_evidence"] is True

            privacy_probe = _payload(
                await session.call_tool("rag_search", {"query": "私人日记", "paths": ["95_Private"]})
            )
            assert privacy_probe["hits"] == []

            note = _payload(await session.call_tool("read_note", {"note": "10_Notes/设备清单.md"}))
            assert "望远镜" in note["content"]
            assert len(note["hash"]) == 64

            listed = _payload(await session.call_tool("list_notes", {"pattern": "设备"}))
            assert any(item["note"] == "10_Notes/设备清单.md" for item in listed["notes"])

    asyncio.run(run())
