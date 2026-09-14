from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from effective_models.prepare_local_artifacts import copy_verified, sha256_file


class VerifiedCopyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.bin"
        self.destination = self.root / "nested" / "destination.bin"
        self.payload = (b"artifact-contract-test\n" * 1024) + b"end"
        self.source.write_bytes(self.payload)
        self.expected = hashlib.sha256(self.payload).hexdigest().upper()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_copy_is_verified_and_source_is_preserved(self) -> None:
        self.assertEqual(copy_verified(self.source, self.destination, self.expected), "copied")
        self.assertEqual(self.source.read_bytes(), self.payload)
        self.assertEqual(self.destination.read_bytes(), self.payload)
        self.assertEqual(sha256_file(self.destination), self.expected)

    def test_matching_existing_destination_is_skipped(self) -> None:
        self.destination.parent.mkdir(parents=True)
        self.destination.write_bytes(self.payload)
        self.assertEqual(copy_verified(self.source, self.destination, self.expected), "skipped")

    def test_different_existing_destination_is_never_overwritten(self) -> None:
        self.destination.parent.mkdir(parents=True)
        self.destination.write_bytes(b"user-owned-different-content")
        before = self.destination.read_bytes()
        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            copy_verified(self.source, self.destination, self.expected)
        self.assertEqual(self.destination.read_bytes(), before)

    def test_bad_source_hash_stops_before_destination_creation(self) -> None:
        with self.assertRaisesRegex(ValueError, "Source SHA-256 mismatch"):
            copy_verified(self.source, self.destination, "0" * 64)
        self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
