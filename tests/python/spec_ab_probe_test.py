"""Offline checks for the paired drafter probe's measurements and corpus."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import spec_ab_probe_dgpp as probe
from spec_ab_probe_prompts import CELLS, CODE


def snapshot(rounds=0, accepted=0, prefills=0, active=0):
    return {"active": active, "queued": 0, "prompts_prefilled": prefills,
            "spec_decode": {"num_draft_tokens_total": 7 * rounds,
                            "num_accepted_tokens_total": accepted,
                            "num_drafts_total": rounds,
                            "num_draft_tokens_per_pos_total": [rounds] * 7,
                            "num_accepted_tokens_per_pos_total": [accepted] + [0] * 6}}


class SpecProbeTest(unittest.TestCase):
    def test_tau_includes_bonus_while_alpha_counts_only_drafts(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "results.json"
            # Start with nonzero counters to test deltas, then delay publication
            # of the retired request beyond delivery of the HTTP response.
            before = snapshot(10, 20, 3)
            after = snapshot(20, 40, 4)
            with patch.object(probe, "metrics", side_effect=[before, before, after]), \
                    patch.object(probe.time, "sleep"), \
                    patch.object(probe.urllib.request, "urlopen", return_value=io.BytesIO(b'{"usage": {}}')), \
                    redirect_stdout(io.StringIO()):
                probe.run(CELLS[:1], out, "http://fixture", "fixture")
            row = json.loads(out.read_text())[0]
            self.assertEqual(row["steps"], 10)
            self.assertEqual(row["tau"], 3.0)
            self.assertEqual(row["alpha"], round(20 / 70, 4))

    def test_busy_or_unrelated_traffic_cannot_be_reported_as_one_request(self):
        with patch.object(probe, "metrics", return_value=snapshot(active=1)):
            with self.assertRaisesRegex(RuntimeError, "idle"):
                probe.idle_metrics("http://fixture")
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(probe, "metrics", side_effect=[snapshot(), snapshot(10, 20, 2)]), \
                patch.object(probe.urllib.request, "urlopen", return_value=io.BytesIO(b'{"usage": {}}')):
            with self.assertRaisesRegex(RuntimeError, "exactly one request"):
                probe.run(CELLS[:1], Path(directory) / "out.json", "http://fixture", "fixture")

    def test_model_must_identify_the_metrics_owner(self):
        with patch.object(probe, "served_ids", return_value=["target"]):
            self.assertEqual(probe.resolve_model("http://fixture", None), "target")
            self.assertEqual(probe.resolve_model("http://fixture", "target"), "target")
            with self.assertRaises(ValueError):
                probe.resolve_model("http://fixture", "unknown")
        for ids in ([], ["target", "other"]):
            with self.subTest(ids=ids), patch.object(probe, "served_ids", return_value=ids):
                with self.assertRaises(ValueError):
                    probe.resolve_model("http://fixture", "target")

    def test_corpus_keeps_distinct_tasks_at_each_length(self):
        self.assertEqual(len(CELLS), 13)
        prompts = [c["messages"][0]["content"] for c in CELLS]
        self.assertEqual(len(set(prompts)), len(prompts))
        for offset in (0, 3, 6):
            for i in range(3):
                self.assertIn(CODE[i], prompts[offset + i])


if __name__ == "__main__":
    unittest.main()
