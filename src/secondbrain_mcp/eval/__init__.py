"""Evaluation and shadow-comparison tooling.

Kept intentionally thin: importing this package must not pull in the CLI modules,
so ``python -m secondbrain_mcp.eval.run`` executes a fresh module rather than a
cached import.
"""

__all__: list[str] = []
