"""Numerical model, exports, and optional Bokeh UI for the LRS2 ETC."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits


CHANNELS = ("UV", "Orange", "Red", "Farred")
SKY_KINDS = ("bright", "grey", "dark")

DEFAULT_CHANNEL = "UV"
DEFAULT_SEEING = 1.8
DEFAULT_MAGNITUDE = 19.0
DEFAULT_SKY = "dark"
DEFAULT_EXPTIME = 1200.0

# These constants are the values used by the original Panacea calculator.
AREA = 45e4
FIBER_AREA = 0.452195
COLLECTING_FRACTION = 0.52
READ_NOISE_TERM = 3.0 * np.sqrt(2.0)


@dataclass(frozen=True)
class CalibrationData:
    """Loaded LRS2 calibration arrays."""

    positions_x: np.ndarray
    positions_y: np.ndarray
    wavelengths: Mapping[str, np.ndarray]
    responses: Mapping[str, np.ndarray]
    sky: Mapping[Tuple[str, str], np.ndarray]
    config_dir: Path


def _array_from_fits(path: Path, row: int) -> np.ndarray:
    """Read one row from a two-row FITS primary array and close the file."""

    with fits.open(path, memmap=False) as hdul:
        data = np.asarray(hdul[0].data[row], dtype=float)
    return np.array(data, copy=True)


def _default_config_dir() -> Path:
    """Return the installed/source-tree calibration directory."""

    # lrs2_config is installed beside lrs2_etc and is also present beside it
    # in a source checkout. Resolving the path makes calls independent of cwd.
    return Path(__file__).resolve().parent.parent / "lrs2_config"


@lru_cache(maxsize=8)
def _load_calibration_cached(config_dir: str) -> CalibrationData:
    config = Path(config_dir).expanduser().resolve()
    if not config.is_dir():
        raise FileNotFoundError(f"Calibration directory does not exist: {config}")

    positions_path = config / "positions.fits"
    with fits.open(positions_path, memmap=False) as hdul:
        positions = np.asarray(hdul[0].data, dtype=float)
    if positions.shape[0] < 2:
        raise ValueError(f"Expected x/y positions in {positions_path}")

    wavelengths: Dict[str, np.ndarray] = {}
    responses: Dict[str, np.ndarray] = {}
    sky: Dict[Tuple[str, str], np.ndarray] = {}
    for channel in CHANNELS:
        response_path = config / f"response_{channel.lower()}.fits"
        wavelengths[channel] = _array_from_fits(response_path, 0)
        responses[channel] = _array_from_fits(response_path, 1)
        if wavelengths[channel].shape != responses[channel].shape:
            raise ValueError(f"Wavelength/response shape mismatch for {channel}")
        for sky_kind in SKY_KINDS:
            sky_path = config / f"{sky_kind}_{channel.lower()}_sky.fits"
            sky[(sky_kind, channel)] = _array_from_fits(sky_path, 1)
            if sky[(sky_kind, channel)].shape != wavelengths[channel].shape:
                raise ValueError(f"Sky/response shape mismatch for {sky_kind}/{channel}")

    return CalibrationData(
        positions_x=np.array(positions[0], copy=True),
        positions_y=np.array(positions[1], copy=True),
        wavelengths=wavelengths,
        responses=responses,
        sky=sky,
        config_dir=config,
    )


def load_calibration(config_dir: Optional[Path] = None) -> CalibrationData:
    """Load all calibration FITS files, independent of the caller's cwd."""

    directory = Path(config_dir) if config_dir is not None else _default_config_dir()
    return _load_calibration_cached(str(directory.expanduser().resolve()))


def _validate_channel(channel: str) -> None:
    if channel not in CHANNELS:
        choices = ", ".join(CHANNELS)
        raise ValueError(f"Unknown channel {channel!r}; choose one of {choices}")


def _validate_sky_kind(sky_kind: str) -> None:
    if sky_kind not in SKY_KINDS:
        choices = ", ".join(SKY_KINDS)
        raise ValueError(f"Unknown sky kind {sky_kind!r}; choose one of {choices}")


def calculate_sn(
    channel: str,
    seeing: float,
    magnitude: float,
    sky_kind: str,
    exptime: float,
    calibration: Optional[CalibrationData] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate wavelength, S/N per pixel, and noise for one channel.

    The spatial extraction weights are applied consistently to both signal and
    variance. In particular, the variance sum uses ``weights**2``; this is
    the weighting used by the Bokeh callback as well as this Python path.
    """

    calibration = calibration or load_calibration()
    _validate_channel(channel)
    _validate_sky_kind(sky_kind)
    seeing = float(seeing)
    magnitude = float(magnitude)
    exptime = float(exptime)
    if seeing <= 0:
        raise ValueError("Seeing must be positive")
    if exptime <= 0:
        raise ValueError("Exposure time must be positive")

    wave = np.asarray(calibration.wavelengths[channel], dtype=float)
    response = np.asarray(calibration.responses[channel], dtype=float)
    sky_flux = np.asarray(calibration.sky[(sky_kind, channel)], dtype=float)
    if not (np.all(np.isfinite(wave)) and np.all(np.isfinite(response))):
        raise ValueError(f"Non-finite wavelength or response values for {channel}")
    if np.any(response <= 0):
        raise ValueError(f"Response must be positive for {channel}")
    if wave.size < 2 or not np.all(np.isfinite(sky_flux)):
        raise ValueError(f"Invalid calibration arrays for {channel}/{sky_kind}")

    mean_wave = float(np.mean(wave))
    dw = float(np.mean(np.diff(wave)))
    object_flux = 10.0 ** (-0.4 * (magnitude - 23.9)) * 1e-29 * 3e18 / mean_wave**2
    object_counts = object_flux / response * exptime * dw * AREA * COLLECTING_FRACTION
    sky_counts = sky_flux / response * exptime * dw * AREA * FIBER_AREA * COLLECTING_FRACTION

    distance = np.sqrt(calibration.positions_x**2 + calibration.positions_y**2)
    sigma = seeing / 2.35
    weights = np.exp(-0.5 * distance**2 / sigma**2)
    weights /= np.sum(weights)
    mask = distance <= sigma * 2.5

    spatial_weights = weights[:, np.newaxis]
    extraction_mask = mask[:, np.newaxis]
    signal = np.sum(object_counts * spatial_weights**2 * extraction_mask, axis=0)
    variance_per_fiber = READ_NOISE_TERM**2 + object_counts * spatial_weights + sky_counts
    noise = np.sqrt(np.sum(variance_per_fiber * spatial_weights**2 * extraction_mask, axis=0))
    sn_per_pixel = signal / noise
    return wave.copy(), sn_per_pixel, noise


def get_sn(
    chan: str,
    _seeing: float,
    _objmag: float,
    skykind: str,
    _exptime: float,
    calibration: Optional[CalibrationData] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility wrapper retaining the legacy calculator's API."""

    return calculate_sn(
        chan,
        _seeing,
        _objmag,
        skykind,
        _exptime,
        calibration or load_calibration(),
    )


def write_sn_table(
    output_path: Path,
    channel: str,
    seeing: float,
    magnitude: float,
    sky_kind: str,
    exptime: float,
    result: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    calibration: Optional[CalibrationData] = None,
) -> Path:
    """Write a plain ASCII wavelength/S/N/noise table and return its path."""

    calibration = calibration or load_calibration()
    if result is None:
        result = calculate_sn(channel, seeing, magnitude, sky_kind, exptime, calibration)
    wavelength, sn_per_pixel, noise = result
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# LRS2 exposure-time calculator\n")
        stream.write(f"# channel: {channel}\n")
        stream.write(f"# seeing_arcsec: {float(seeing):.6g}\n")
        stream.write(f"# magnitude_AB: {float(magnitude):.6g}\n")
        stream.write(f"# sky: {sky_kind}\n")
        stream.write(f"# exptime_s: {float(exptime):.6g}\n")
        stream.write("wavelength_A  sn_per_pixel  noise\n")
        for values in zip(wavelength, sn_per_pixel, noise):
            stream.write("{:.8f}  {:.8e}  {:.8e}\n".format(*values))
    return path


def _bokeh_callback_code() -> str:
    """Return the browser-side calculation matching :func:`calculate_sn`."""

    return r"""
    const data = source.data;
    const wave = data[channel.value + '_wave'];
    const resp = data[channel.value + '_resp'];
    const sky = data[sky_kind.value + '_' + channel.value + '_sky'];
    const seeing_sigma = seeing.value / 2.35;
    const magnitude_value = magnitude.value;
    const exposure = exptime.value;
    const weights = new Array(px.length);
    const mask = new Array(px.length);
    let weight_sum = 0.0;
    for (let j = 0; j < px.length; j++) {
        const distance = Math.sqrt(px[j] * px[j] + py[j] * py[j]);
        weights[j] = Math.exp(-0.5 * distance * distance /
                              (seeing_sigma * seeing_sigma));
        mask[j] = distance <= seeing_sigma * 2.5 ? 1.0 : 0.0;
        weight_sum += weights[j];
    }
    for (let j = 0; j < weights.length; j++) {
        weights[j] /= weight_sum;
    }
    let mean_wave = 0.0;
    for (let i = 0; i < wave.length; i++) {
        mean_wave += wave[i];
    }
    mean_wave /= wave.length;
    let dw = 0.0;
    for (let i = 1; i < wave.length; i++) {
        dw += wave[i] - wave[i - 1];
    }
    dw /= wave.length - 1;
    const object_flux = Math.pow(10.0, -0.4 * (magnitude_value - 23.9)) *
        1E-29 * 3E18 / (mean_wave * mean_wave);
    const x = data['x'];
    const y = data['y'];
    for (let i = 0; i < wave.length; i++) {
        x[i] = wave[i];
        const object_counts = object_flux / resp[i] * exposure * dw * 45E4 * 0.52;
        const sky_counts = sky[i] / resp[i] * exposure * dw * 45E4 * 0.452195 * 0.52;
        let signal = 0.0;
        let variance = 0.0;
        for (let j = 0; j < weights.length; j++) {
            signal += object_counts * weights[j] * weights[j] * mask[j];
            variance += (Math.pow(3.0 * Math.sqrt(2.0), 2) +
                         object_counts * weights[j] + sky_counts) *
                         weights[j] * weights[j] * mask[j];
        }
        y[i] = signal / Math.sqrt(variance);
    }
    let average = 0.0;
    for (let i = 0; i < y.length; i++) {
        average += y[i];
    }
    average /= y.length;
    plot.title.text = 'Average S/N ' + average.toFixed(2);
    plot.x_range.start = wave[0];
    plot.x_range.end = wave[wave.length - 1];
    source.change.emit();
    """


def create_bokeh_app(
    channel: str = DEFAULT_CHANNEL,
    seeing: float = DEFAULT_SEEING,
    magnitude: float = DEFAULT_MAGNITUDE,
    sky_kind: str = DEFAULT_SKY,
    exptime: float = DEFAULT_EXPTIME,
    calibration: Optional[CalibrationData] = None,
):
    """Build the interactive Bokeh layout without writing an HTML file.

    Returns ``(layout, source, plot)`` so callers and tests can inspect the
    initial numerical curve before saving or embedding the application.
    """

    try:
        from bokeh.layouts import column, row
        from bokeh.models import ColumnDataSource, CustomJS, HoverTool, PrintfTickFormatter
        from bokeh.models import Select, Slider
        from bokeh.plotting import figure
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Bokeh is required for the interactive calculator; install lrs2-etc[bokeh]") from exc

    _validate_channel(channel)
    _validate_sky_kind(sky_kind)
    calibration = calibration or load_calibration()
    wavelength, sn_per_pixel, noise = calculate_sn(
        channel, seeing, magnitude, sky_kind, exptime, calibration
    )
    source_data = {"x": wavelength, "y": sn_per_pixel, "noise": noise}
    for item in CHANNELS:
        source_data[f"{item}_wave"] = calibration.wavelengths[item]
        source_data[f"{item}_resp"] = calibration.responses[item]
        for kind in SKY_KINDS:
            source_data[f"{kind}_{item}_sky"] = calibration.sky[(kind, item)]
    source = ColumnDataSource(data=source_data)

    plot = figure(
        height=600,
        width=800,
        title=f"Exposure Time Calculator (Average S/N {np.mean(sn_per_pixel):.2f})",
        tools="crosshair, pan, reset, save, wheel_zoom",
        x_range=(float(wavelength.min()), float(wavelength.max())),
        y_range=(0.01, 500),
        y_axis_type="log",
    )
    plot.line("x", "y", source=source, line_width=3, line_alpha=0.6, line_color="salmon")
    plot.xaxis.major_label_text_font_size = "16pt"
    plot.yaxis.major_label_text_font_size = "16pt"
    plot.xaxis.axis_label = "Wavelength (A)"
    plot.xaxis.axis_label_text_font_size = "20pt"
    plot.yaxis.axis_label = "S/N per pixel"
    plot.yaxis.axis_label_text_font_size = "20pt"
    plot.xaxis.major_tick_line_color = "firebrick"
    plot.xaxis.major_tick_line_width = 3
    plot.xaxis.minor_tick_line_color = "orange"
    plot.yaxis.major_tick_line_color = "firebrick"
    plot.yaxis.major_tick_line_width = 3
    plot.yaxis.minor_tick_line_color = "orange"
    plot.yaxis[0].formatter = PrintfTickFormatter(format="%3.2f")
    plot.add_tools(
        HoverTool(
            tooltips=[("Wavelength", "@x{0.2f}"), ("S/N per Pixel", "@y{0.2f}")],
            formatters={"Wavelength": "numeral", "S/N per Pixel": "numeral"},
            mode="mouse",
        )
    )

    callback = CustomJS(
        args={
            "source": source,
            "plot": plot,
            "px": calibration.positions_x,
            "py": calibration.positions_y,
        },
        code=_bokeh_callback_code(),
    )
    channel_select = Select(value=channel, title="Channel", options=list(CHANNELS))
    sky_select = Select(value=sky_kind, title="Sky Background", options=list(SKY_KINDS))
    seeing_slider = Slider(title="Seeing (arcsec)", value=seeing, start=0.8, end=4.0, step=0.1)
    magnitude_slider = Slider(title="Magnitude (AB)", value=magnitude, start=6.0, end=25.0, step=0.1)
    exptime_slider = Slider(title="Exposure Time (s)", value=exptime, start=5.0, end=3600.0, step=5.0)
    callback.args.update(
        channel=channel_select,
        sky_kind=sky_select,
        seeing=seeing_slider,
        magnitude=magnitude_slider,
        exptime=exptime_slider,
    )
    for widget in (channel_select, sky_select, seeing_slider, magnitude_slider, exptime_slider):
        widget.js_on_change("value", callback)
    inputs = column(channel_select, sky_select, seeing_slider, magnitude_slider, exptime_slider)
    return row(inputs, plot, width=1200), source, plot


def save_bokeh(
    output_path: Path,
    channel: str = DEFAULT_CHANNEL,
    seeing: float = DEFAULT_SEEING,
    magnitude: float = DEFAULT_MAGNITUDE,
    sky_kind: str = DEFAULT_SKY,
    exptime: float = DEFAULT_EXPTIME,
    calibration: Optional[CalibrationData] = None,
) -> Path:
    """Build and save the interactive calculator as a standalone HTML file."""

    from bokeh.plotting import output_file, save

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    layout, _, _ = create_bokeh_app(channel, seeing, magnitude, sky_kind, exptime, calibration)
    output_file(str(path), title="LRS2 Exposure Time Calculator")
    save(layout)
    return path


def _text_output_paths(output: Path, channels: Iterable[str]) -> Dict[str, Path]:
    channels = tuple(channels)
    if len(channels) == 1:
        return {channels[0]: output}
    if (output.exists() and output.is_dir()) or output.suffix == "":
        return {channel: output / f"{channel.lower()}_sn.txt" for channel in channels}
    suffix = output.suffix or ".txt"
    return {channel: output.parent / f"{output.stem}_{channel.lower()}{suffix}" for channel in channels}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculate LRS2 exposure-time S/N curves.")
    parser.add_argument("--channel", choices=CHANNELS, default=DEFAULT_CHANNEL)
    parser.add_argument("--all-channels", action="store_true", help="Export text tables for every channel.")
    parser.add_argument("--seeing", type=float, default=DEFAULT_SEEING, help="Seeing FWHM in arcsec.")
    parser.add_argument("--magnitude", "--ab-magnitude", dest="magnitude", type=float, default=DEFAULT_MAGNITUDE)
    parser.add_argument("--sky", choices=SKY_KINDS, default=DEFAULT_SKY)
    parser.add_argument("--exptime", type=float, default=DEFAULT_EXPTIME, help="Exposure time in seconds.")
    parser.add_argument("--sn-output", "--output", dest="sn_output", type=Path, help="ASCII S/N table path.")
    parser.add_argument("--bokeh-output", type=Path, default=Path("lrs2_exptime_calc.html"))
    parser.add_argument("--no-bokeh", action="store_true", help="Do not create the interactive HTML output.")
    parser.add_argument("--config-dir", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.no_bokeh and args.sn_output is None:
        raise SystemExit("--no-bokeh requires --sn-output or --output")

    calibration = load_calibration(args.config_dir)
    channels = CHANNELS if args.all_channels else (args.channel,)
    if args.sn_output is not None:
        for channel, output in _text_output_paths(args.sn_output, channels).items():
            result = calculate_sn(channel, args.seeing, args.magnitude, args.sky, args.exptime, calibration)
            write_sn_table(
                output,
                channel,
                args.seeing,
                args.magnitude,
                args.sky,
                args.exptime,
                result=result,
                calibration=calibration,
            )
            print(f"Wrote {output}")

    if not args.no_bokeh:
        save_bokeh(
            args.bokeh_output,
            args.channel,
            args.seeing,
            args.magnitude,
            args.sky,
            args.exptime,
            calibration,
        )
        print(f"Wrote {args.bokeh_output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
