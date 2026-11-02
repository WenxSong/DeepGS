from pathlib import Path

import pytest

from deepgs.data import load_profile_txt

ROOT = Path(__file__).resolve().parent.parent
TH_TXT = ROOT / "intermediate" / "data" / "th.txt"
TRUE_TERMS = {"theta_z", "theta_zz", "theta_z^2"}


@pytest.fixture(scope="session")
def loam():
    """The upper 30 cm of the loam record; this keeps the finite-difference libraries small."""
    if not TH_TXT.exists():
        pytest.skip("Optional local loam research data are not distributed with the source package")
    full = load_profile_txt(TH_TXT)
    return full.select_depths(full.depths[:31])
