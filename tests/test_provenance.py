import json
from pathlib import Path

import pytest

from deepgs.cli import main
from deepgs.config import GSRConfig
from deepgs.provenance import file_record


def test_file_hash_tracks_content(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("before")
    first = file_record(path)
    path.write_text("after")
    assert first["sha256"] != file_record(path)["sha256"]


@pytest.mark.parametrize("fail", [False, True])
def test_cli_records_effective_config_and_run_outcome(tmp_path, monkeypatch, fail):
    config = tmp_path / "input.yaml"
    config.write_text("pipeline: gsr\ndata:\n  path: input.txt\n")
    data = tmp_path / "input.txt"
    data.write_text("0 0.1 0.2")
    monkeypatch.chdir(tmp_path)

    def runner(config, output, device):
        assert device == "cpu"
        if fail:
            raise RuntimeError("controlled failure")

    monkeypatch.setitem(__import__("deepgs.cli", fromlist=["RUNNERS"]).RUNNERS, "gsr", runner)
    args = [str(config), "--output", str(tmp_path / "run"), "--device", "cpu"]
    if fail:
        with pytest.raises(RuntimeError, match="controlled failure"):
            main(args)
    else:
        main(args)
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["status"] == ("failed" if fail else "completed")
    assert manifest["configuration"]["sgtr"]["epsilon"] == GSRConfig().sgtr.epsilon
    assert manifest["inputs"][0]["sha256"] == file_record(data)["sha256"]
    with pytest.raises(SystemExit):
        main(args)
