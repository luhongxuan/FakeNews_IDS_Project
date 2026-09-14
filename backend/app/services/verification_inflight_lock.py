"""Cross-process in-flight lock for expensive claim verification.

The verification cache row may not exist yet, so row locks cannot protect the
first verification of a cluster.  PostgreSQL session-level advisory locks can:
all scheduler processes derive the same signed 64-bit key from the cluster id,
and only the winner performs the external search/LLM call.

The caller must keep the lock until the VerificationReport cache write has
committed.  Releasing it immediately after the external call would leave a
race window in which another scheduler could acquire the lock before the new
cache row becomes visible and perform the same expensive verification again.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

LOCK_NAMESPACE = "misinfo-verification-v1"


def advisory_lock_key(resource_id: str, namespace: str = LOCK_NAMESPACE) -> int:
    """Return a deterministic PostgreSQL-compatible signed bigint key."""
    payload = f"{namespace}\0{resource_id}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big", signed=True)


@dataclass
class VerificationInFlightLock:
    """A non-blocking, explicitly released PostgreSQL advisory lock."""

    session: Session
    resource_id: str
    acquired: bool = False

    @property
    def key(self) -> int:
        return advisory_lock_key(self.resource_id)

    def try_acquire(self) -> bool:
        if self.acquired:
            return True
        self.acquired = bool(
            self.session.execute(
                text("SELECT pg_try_advisory_lock(:lock_key)"),
                {"lock_key": self.key},
            ).scalar()
        )
        return self.acquired

    def release(self) -> None:
        if not self.acquired:
            return
        released = bool(
            self.session.execute(
                text("SELECT pg_advisory_unlock(:lock_key)"),
                {"lock_key": self.key},
            ).scalar()
        )
        self.acquired = False
        if not released:
            raise RuntimeError(f"PostgreSQL advisory lock was not held: {self.resource_id}")

