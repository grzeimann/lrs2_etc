from pathlib import Path

import numpy as np
import pytest

from lrs2_etc.calculator import (
    calculate_sn,
    create_bokeh_app,
    load_calibration,
    write_sn_table,
)


@pytest.fixture(scope="module")
def calibration():
    return load_calibration()


def test_shape_and_finite_output(calibration):
    wave, sn, noise = calculate_sn("UV", 1.8, 19.0, "dark", 1200.0, calibration)
    assert wave.shape == sn.shape == noise.shape
    assert wave.size > 0
    assert np.isfinite(wave).all()
    assert np.isfinite(sn).all()
    assert np.isfinite(noise).all()
    assert np.all(noise > 0)


def test_channel_and_sky_validation(calibration):
    with pytest.raises(ValueError, match="Unknown channel"):
        calculate_sn("nope", 1.8, 19.0, "dark", 1200.0, calibration)
    with pytest.raises(ValueError, match="Unknown sky kind"):
        calculate_sn("UV", 1.8, 19.0, "nope", 1200.0, calibration)


def test_exposure_time_increases_signal_to_noise(calibration):
    _, sn_short, _ = calculate_sn("Red", 1.8, 19.0, "dark", 300.0, calibration)
    _, sn_long, _ = calculate_sn("Red", 1.8, 19.0, "dark", 1200.0, calibration)
    assert np.all(sn_long > sn_short)


def test_text_export_matches_calculation(tmp_path: Path, calibration):
    result = calculate_sn("Orange", 1.8, 19.0, "grey", 1200.0, calibration)
    path = write_sn_table(tmp_path / "orange.txt", "Orange", 1.8, 19.0, "grey", 1200.0, result, calibration)
    lines = path.read_text().splitlines()
    assert lines[0].startswith("# LRS2")
    assert lines[6] == "wavelength_A  sn_per_pixel  noise"
    exported = np.loadtxt(path, comments="#", skiprows=7)
    np.testing.assert_allclose(exported[:, 0], result[0], rtol=0, atol=1e-7)
    np.testing.assert_allclose(exported[:, 1:], np.column_stack(result[1:]), rtol=1e-7)


def test_bokeh_initial_curve_uses_same_numerical_path(calibration):
    pytest.importorskip("bokeh")
    _, source, _ = create_bokeh_app("UV", 1.8, 19.0, "dark", 1200.0, calibration)
    _, expected, _ = calculate_sn("UV", 1.8, 19.0, "dark", 1200.0, calibration)
    np.testing.assert_allclose(source.data["y"], expected)
