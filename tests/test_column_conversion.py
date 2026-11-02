"""Normalization preserves measurements and makes the sampling clock explicit."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from deepgs.datasets import validate_dataset, load_standard_csv, load_standard_npz


def converter():
    path = Path(__file__).resolve().parents[1] / "examples/convert_soil_column.py"
    spec = importlib.util.spec_from_file_location("column_converter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.convert


def test_conversion_keeps_values_columns_and_explicit_minutes(tmp_path):
    source = tmp_path / "original.csv"
    content = b"0.05m,0.85m\n0.203,0.342\n0.205,0.341\n0.209,0.340\n"
    source.write_bytes(content)
    output = tmp_path / "case"
    converter()(source, output, minutes_per_row=1.)
    report = validate_dataset(output)
    assert report["kind"] == "real" and report["shape"] == [3, 2]
    npz, csv = load_standard_npz(output / "observations.npz"), load_standard_csv(output / "observations.csv")
    np.testing.assert_array_equal(npz.theta, np.array([[.203, .342], [.205, .341], [.209, .340]]))
    for key in ("times", "depths", "theta"):
        np.testing.assert_array_equal(getattr(npz, key), getattr(csv, key))
    np.testing.assert_array_equal(npz.times, np.arange(3) / 1440)
    assert source.read_bytes() == content
    metadata = yaml.safe_load((output / "metadata.yaml").read_text())
    assert metadata["license"] == "not_specified"
    assert metadata["details"]["sampling_interval_minutes"] == 1.
    with pytest.raises(FileExistsError):
        converter()(source, output, minutes_per_row=1.)


@pytest.mark.parametrize("interval", [0., -1., float("nan")])
def test_conversion_requires_explicit_valid_interval(tmp_path, interval):
    source = tmp_path / "original.csv"
    source.write_text("0.05m,0.85m\n.2,.3\n.21,.31\n")
    with pytest.raises(ValueError, match="positive"):
        converter()(source, tmp_path / "out", minutes_per_row=interval)
    assert not (tmp_path / "out").exists()


def test_conversion_rejects_missing_values(tmp_path):
    source = tmp_path / "original.csv"
    source.write_text("0.05m,0.85m\n.2,.3\n,.31\n")
    with pytest.raises(ValueError, match="missing"):
        converter()(source, tmp_path / "out", minutes_per_row=1.)
    assert not (tmp_path / "out").exists()
