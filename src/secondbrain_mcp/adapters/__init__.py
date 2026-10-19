"""Adapters: MCP stdio and CLI.

Deliberately empty of imports. ``python -m secondbrain_mcp.adapters.mcp_stdio``
executes this module from a fresh namespace; re-exporting it here would make the
module appear twice in ``sys.modules`` and print a RuntimeWarning on every
server start.
"""

__all__: list[str] = []
