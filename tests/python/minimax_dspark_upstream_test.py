"""Guard matched-prefix acceptance accounting and thinking/answer boundaries."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from minimax_dspark_upstream import phase, prefix_length, summarize


class UpstreamAcceptanceTest(unittest.TestCase):
    def test_changed_prefix_and_unobserved_tail(self):
        rows = [
            dict(
                case="code",
                proposals=[1, 9, 3],
                upstream=[1, 2, 3],
                truth=[1, 2, 3],
                phases=["thinking"] * 3,
            ),
            dict(
                case="code",
                proposals=[9, 2, 3],
                upstream=[1, 2, 8],
                truth=[1, 2, 3],
                phases=["thinking"] * 3,
            ),
            dict(
                case="code",
                proposals=[1, 2, 3],
                upstream=[1, 2, 3],
                truth=[1],
                phases=["thinking"],
            ),
        ]
        result = summarize(rows)["code"]["all"]
        # Matching a later token does not rescue an earlier rejected proposal.
        self.assertEqual(result["proposals"][2]["accepted"], 0)
        self.assertEqual(result["proposals"][2]["reached"], 0)
        # The independent continuation supplies valid labels after a changed
        # proposal. A truncated continuation contributes no invented labels.
        self.assertEqual(result["upstream"][2]["eligible"], 2)
        self.assertEqual(result["upstream"][2]["accepted"], 1)
        self.assertEqual(result["upstream"][2]["conditional_pct"], 50)
        self.assertEqual(prefix_length([1, 2, 3], [1]), 1)

    def test_phase_is_generated_position_not_prompt_class(self):
        start, end = 100, 101
        ids = [1, start, 2, end, 3]
        self.assertEqual(phase(ids[:3], start, end), "thinking")
        self.assertEqual(phase(ids, start, end), "answer")
        self.assertEqual(phase(ids + [start], start, end), "thinking")
        self.assertEqual(phase([1, 2], start, end), "unmarked")
        rows = [
            dict(
                case="json",
                proposals=[1, 2],
                upstream=[1, 2],
                truth=[1, 2],
                phases=["thinking", "answer"],
            )
        ]
        result = summarize(rows)["json"]
        self.assertEqual(result["thinking"]["upstream"][0]["eligible"], 1)
        self.assertEqual(result["thinking"]["upstream"][1]["eligible"], 0)
        self.assertEqual(result["answer"]["upstream"][1]["accepted"], 1)


if __name__ == "__main__":
    unittest.main()
