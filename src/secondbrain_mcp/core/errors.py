"""Error contract.

The MCP SDK maps any plain exception to ``"Error executing tool X"``, discarding
the message. The previous system's carefully-worded recovery hints ("retry with
the current content hash") therefore never reached the model — the entire
optimistic-concurrency contract was invisible over the wire.

Every failure the caller can act on is a ``GatewayError`` carrying a stable code
and an explicit recovery instruction; the adapter is responsible for rendering
both onto the wire.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GatewayError(Exception):
    code: str
    message: str
    recovery: str | None = None
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        text = f"[{self.code}] {self.message}"
        if self.recovery:
            text = f"{text} — {self.recovery}"
        return text

    def to_payload(self) -> dict:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "recovery": self.recovery,
                **({"details": self.details} if self.details else {}),
            }
        }


class PathDenied(GatewayError):
    def __init__(self, message: str = "Path is outside the permitted area"):
        super().__init__("PATH_DENIED", message, "Use a vault-relative path inside a readable zone")


class PrivateDenied(GatewayError):
    def __init__(self, message: str = "Private notes are not readable through this gateway"):
        super().__init__("PRIVATE_DENY", message, "Choose a note outside the private zone")


class HashConflict(GatewayError):
    def __init__(self, message: str = "Note changed since it was read"):
        super().__init__(
            "HASH_CONFLICT",
            message,
            "Call note_read again and pass the hash it returns as expected_hash",
        )


class LockBusy(GatewayError):
    def __init__(self, message: str = "Note is being modified by another agent"):
        super().__init__("LOCK_BUSY", message, "Retry in a few seconds")


class HashRequired(GatewayError):
    def __init__(self) -> None:
        super().__init__(
            "HASH_REQUIRED",
            "expected_hash is required for every mutating operation",
            "Call note_read first and pass the hash it returns",
        )


class NoteNotFound(GatewayError):
    def __init__(self, note: str) -> None:
        super().__init__("NOTE_NOT_FOUND", f"Note does not exist: {note}", "Check the path with note_list")


class InvalidArguments(GatewayError):
    def __init__(self, message: str, recovery: str | None = None):
        super().__init__("INVALID_ARGUMENTS", message, recovery)


class IndexUnavailable(GatewayError):
    def __init__(self, message: str) -> None:
        super().__init__("INDEX_UNAVAILABLE", message, "Run index_rebuild")


class ModelMismatch(GatewayError):
    def __init__(self, stored: str, active: str) -> None:
        super().__init__(
            "INDEX_MODEL_MISMATCH",
            f"Index was built with {stored} but the active embedder is {active}",
            "Run index_rebuild --force to rebuild vectors for the active model",
        )


class RestUnavailable(GatewayError):
    def __init__(self, message: str = "Obsidian Local REST API is not reachable") -> None:
        super().__init__(
            "REST_UNAVAILABLE",
            message,
            "Start Obsidian and enable the Local REST API plugin, then retry",
        )
