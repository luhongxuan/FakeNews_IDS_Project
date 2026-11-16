import json
import tempfile
import unittest
from pathlib import Path

from app.services.pheme_graph_builder import build_graph_from_pheme


class PhemeGraphBuilderTest(unittest.TestCase):
    def _write_tweet(self, path: Path, tweet_id: str, user: str, created_at: str) -> None:
        path.write_text(
            json.dumps(
                {
                    "id_str": tweet_id,
                    "created_at": created_at,
                    "user": {"screen_name": user},
                }
            ),
            encoding="utf-8",
        )

    def test_builds_user_edges_and_metadata_without_repository_root_import(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            thread = base / "demo-all-rnr-threads" / "rumours" / "thread-1"
            source = thread / "source-tweets"
            reactions = thread / "reactions"
            source.mkdir(parents=True)
            reactions.mkdir()
            (thread.parent / "._dataset-sidecar").write_text("", encoding="utf-8")

            self._write_tweet(
                source / "root.json",
                "root",
                "source_user",
                "Mon Jan 05 10:00:00 +0000 2015",
            )
            self._write_tweet(
                reactions / "reply.json",
                "reply",
                "reply_user",
                "Mon Jan 05 10:01:00 +0000 2015",
            )
            (thread / "structure.json").write_text(
                json.dumps({"root": {"reply": []}}), encoding="utf-8"
            )

            graph, first_seen, rumour_count = build_graph_from_pheme(str(base), "demo")

            self.assertEqual(first_seen, "2015-01-05")
            # Historical backend semantics count every entry in the rumours
            # directory, including PHEME sidecars; dependency localization
            # must not silently change the API field.
            self.assertEqual(rumour_count, 2)
            self.assertEqual(list(graph.edges()), [("source_user", "reply_user")])
            self.assertEqual(graph.edges["source_user", "reply_user"]["thread_id"], "thread-1")
            self.assertTrue(graph.edges["source_user", "reply_user"]["is_rumour"])

    def test_missing_event_returns_empty_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            graph, first_seen, rumour_count = build_graph_from_pheme(temp_dir, "missing")

            self.assertEqual(graph.number_of_nodes(), 0)
            self.assertIsNone(first_seen)
            self.assertEqual(rumour_count, 0)


if __name__ == "__main__":
    unittest.main()
