"""Exercise discovery with config metadata only: no weights, CUDA, or peers."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    binary, fixture = sys.argv[1:]
    with tempfile.TemporaryDirectory(prefix="dgpp-heretic-") as tmp:
        model = Path(tmp) / 'model with "quotes"'
        model.mkdir()
        (model / "config.json").write_text(Path(fixture).read_text())
        config = Path(tmp) / "cluster.json"
        cluster = {"model": str(model), "nodes": ["rank0", "rank1", "rank2", "rank3"]}

        def run(rank="0"):
            config.write_text(json.dumps(cluster))
            return subprocess.run(
                [binary, "--config", str(config), "--rank", rank],
                capture_output=True, text=True, timeout=15,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            )

        for flag in (None, False):
            cluster["engine"] = {} if flag is None else {"heretic_research": flag}
            result = run()
            assert result.returncode != 0 and not result.stdout
            assert "engine.heretic_research=true" in result.stderr, result.stderr

        cluster["engine"] = {"heretic_research": True}
        all_ranks = []
        for rank in range(4):
            result = run(str(rank))
            assert result.returncode == 0, result.stderr
            data = json.loads(result.stdout)
            assert data["schema"] == "dgpp.heretic.targets" and data["version"] == 1
            assert data["capabilities"] == ["describe_targets"]
            assert data["checkpoint"] == str(model) and data["model"] == str(model)
            assert data["world"] == 4 and data["rank"] == rank
            assert data["layer_count"] == 5 and len(data["targets"]) == 10
            all_ranks.append(data["targets"])
        for i, target in enumerate(all_ranks[0]):
            assert target["input_begin"] == 0
            assert all_ranks[-1][i]["input_end"] == target["input_features"]
            for rank in range(3):
                assert all_ranks[rank][i]["input_end"] == all_ranks[rank + 1][i]["input_begin"]

        for rank in ("-1", "4", "0junk", "99999999999999999999"):
            result = run(rank)
            assert result.returncode != 0 and "--rank" in result.stderr, result.stderr

        cluster["model"] = str(Path(tmp) / "missing-checkpoint")
        cluster["engine"]["heretic_research"] = False
        result = run()
        assert "research is disabled" in result.stderr, "flag must be checked before checkpoint access"

    print("Heretic discovery: opt-in, metadata-only loading, JSON, four rank slices, and errors passed")


if __name__ == "__main__":
    main()
