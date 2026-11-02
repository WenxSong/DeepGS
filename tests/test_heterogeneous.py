"""Heterogeneous reference coefficients and the soil option of the deepgs pipeline."""
import numpy as np
import pytest

from deepgs.config import DataConfig, ExtendedDeepGSConfig, FieldConfig, GroupConfig, LibraryConfig, ReferenceSoilConfig, SGTRConfig, load_config
from deepgs.data import load_profile_txt
from deepgs.grouping import build_system
from deepgs.library import finite_difference_library
from deepgs.physics import SOILS, HeterogeneousSoil, coefficient_errors, load_heterogeneous_soil
from deepgs.pipelines import run_extended_deepgs
from deepgs.sparse import select_sgtr

from conftest import ROOT

FOLDER = ROOT / "intermediate" / "data" / "loam_all_0.2"
FOUR_TERMS = {"1", "theta_z", "theta_zz", "theta_z^2"}


@pytest.fixture(scope="module")
def soil():
    if not (FOLDER / "ks.npy").exists():
        pytest.skip("Optional local heterogeneous parameter arrays are not distributed")
    return load_heterogeneous_soil(FOLDER, theta_s=0.43, theta_r=0.078)


@pytest.fixture(scope="module")
def observations():
    if not (FOLDER / "th.txt").exists():
        pytest.skip("Optional local heterogeneous research profile is not distributed")
    return load_profile_txt(FOLDER / "th.txt")


def uniform(**changes):
    values = dict(Ks=0.2496, alpha=3.6, n=1.56, theta_s=0.43, theta_r=0.078)
    values.update(changes)
    depths = np.linspace(0, 1, 101)
    return HeterogeneousSoil(depths, **{k: np.full(101, v) if isinstance(v, float) else v for k, v in values.items()})


def test_uniform_soil_reduces_to_homogeneous_equation():
    theta, z = np.linspace(0.15, 0.40, 20), np.full(20, 0.5)
    coefficients = uniform().coefficients(theta, z)
    reference = SOILS["loam"].coefficients(theta)
    assert coefficients["1"] == pytest.approx(np.zeros(20), abs=1e-9)
    for name, values in reference.items():
        assert coefficients[name] == pytest.approx(values, rel=1e-9)


def test_varying_theta_s_and_theta_r_matches_flux_divergence():
    """f0 + f1 theta_z + f2 theta_zz + f3 theta_z^2 must equal the divergence of the flux for any profile."""
    z = np.linspace(0.2, 0.8, 601)
    depths = np.linspace(0, 1, 1001)
    soil = HeterogeneousSoil(depths, 0.25 + 0.05 * np.sin(4 * depths), 3.6 + 0.4 * np.cos(3 * depths), 1.56 + 0.1 * np.sin(5 * depths),
                             0.43 + 0.02 * np.sin(6 * depths), 0.078 + 0.01 * np.cos(4 * depths))
    theta = 0.25 + 0.05 * np.sin(5 * z)
    theta_z, theta_zz = 0.25 * np.cos(5 * z), -1.25 * np.sin(5 * z)

    h = 1e-3
    psi = lambda zz: soil.at(zz).psi(0.25 + 0.05 * np.sin(5 * zz))
    K = lambda zz: soil.at(zz).K(0.25 + 0.05 * np.sin(5 * zz))
    q = lambda zz: K(zz) * ((psi(zz + h) - psi(zz - h)) / (2 * h) - 1)
    divergence = (q(z + h) - q(z - h)) / (2 * h)
    c = soil.coefficients(theta, z)
    assembled = c["1"] + c["theta_z"] * theta_z + c["theta_zz"] * theta_zz + c["theta_z^2"] * theta_z**2
    inner = (z > 0.25) & (z < 0.75)
    assert np.linalg.norm(assembled[inner] - divergence[inner]) / np.linalg.norm(divergence[inner]) < 1e-3


def test_reference_closes_the_simulated_field(soil, observations):
    t, U = observations.times, observations.theta
    Ut = np.gradient(U, t, axis=0)
    Uz = np.gradient(U, 0.01, axis=1)
    Uzz = np.gradient(Uz, 0.01, axis=1)
    index = np.arange(5, 90)
    theta, z = U[::9, :][:, index], np.tile(observations.depths[index], (U[::9].shape[0], 1))
    c = soil.coefficients(theta, z)
    uz, uzz = Uz[::9, :][:, index], Uzz[::9, :][:, index]
    prediction = c["1"] + c["theta_z"] * uz + c["theta_zz"] * uzz + c["theta_z^2"] * uz**2
    truth = Ut[::9, :][:, index]
    assert np.linalg.norm(prediction - truth) / np.linalg.norm(truth) < 0.12


def test_one_short_nodal_parameter_is_extended_to_last_node(soil):
    assert soil.Ks.size == soil.alpha.size == soil.n.size == 101
    assert np.ndim(soil.theta_s) == 0


def test_missing_uniform_parameter_is_an_error():
    if not (FOLDER / "ks.npy").exists():
        pytest.skip("This reference loader check requires optional local parameter arrays")
    with pytest.raises(ValueError, match="theta_s"):
        load_heterogeneous_soil(FOLDER)


@pytest.mark.parametrize("win_size", [0.001, 0.01])
def test_energy_gain_recovers_four_terms_at_two_fd_window_sizes(observations, win_size):
    """A bounded FD check, not a claim of grouping-independent recovery."""
    window = observations.select_depths(observations.depths[3:28])
    library = finite_difference_library(window.theta.T, window.depths, window.times)
    rows = np.isin(np.round(library.depth[:, 0], 6), np.round(window.depths[2:23:2], 6))
    library.theta, library.terms, library.theta_t, library.depth = library.theta[rows], library.terms[rows], library.theta_t[rows], library.depth[rows]
    system = build_system(library, win_size, 100, "group", by_depth=True, merge_tail=0.01)
    assert set(select_sgtr(system, 100, 1e-2, 1, min_energy_gain=0.002).names) == FOUR_TERMS


def test_coefficient_errors_skip_terms_without_reference():
    errors = coefficient_errors({"a": np.array([1.0, 1.0]), "b": np.ones(2)}, {"a": np.array([1.0, 2.0])})
    assert list(errors) == ["a"]
    assert errors["a"] == pytest.approx(1 / np.sqrt(5))


def test_coefficient_errors_do_not_report_undefined_relative_errors():
    assert coefficient_errors({"a": np.array([1.0])}, {"a": np.array([0.0])}) == {}
    assert coefficient_errors({"a": np.array([np.nan])}, {"a": np.array([1.0])}) == {}


def test_soil_option_selects_the_workflow():
    home = load_config(ROOT / "configs" / "deepgs_homogeneous.yaml")
    hetero = load_config(ROOT / "configs" / "deepgs_heterogeneous.yaml")
    assert home[0] == "deepgs"
    assert hetero[0] == "extended_deepgs"
    assert hetero[1].group.by_depth and hetero[1].group.normalization == "group"
    assert not home[1].group.by_depth


@pytest.mark.parametrize("text,message", [
    ("pipeline: deepgs\nsoil: layered\n", "soil"),
    ("pipeline: gsr\nsoil: heterogeneous\n", "soil"),
])
def test_invalid_soil_option_is_rejected(tmp_path, text, message):
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_heterogeneous_run_reports_errors_against_reference(tmp_path):
    if not (FOLDER / "th.txt").exists():
        pytest.skip("This historical integration check requires the optional local heterogeneous data")
    config = ExtendedDeepGSConfig(
        data=DataConfig(path=str(FOLDER / "th.txt"), depths=[0.05, 0.15, 0.25]),
        reference=ReferenceSoilConfig(path=str(FOLDER), theta_s=0.43, theta_r=0.078),
        field=FieldConfig(hidden_layers=2, width=10, iterations=5, activation="sin", normalize_inputs=False),
        library=LibraryConfig(depth_count=3, time_count=300),
        group=GroupConfig(min_group_size=20, normalization="group", by_depth=True),
        sgtr=SGTRConfig(num_tols=10, lam=1e-2, epsilon=10**1.2, corrected=False),
        theta_hidden=[5, 5], depth_hidden=[3], collocation_points=30, train_steps=2, log_every=1)
    result = run_extended_deepgs(config, tmp_path, "cpu")
    assert result["coefficient_errors"] is not None
    assert (tmp_path / "training.json").exists()
