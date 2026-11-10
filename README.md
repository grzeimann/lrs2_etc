# LRS2 exposure-time calculator

This repository contains a small Python 3 exposure-time calculator for the
LRS2 spectrograph. It preserves the interactive Bokeh calculator and can also
export wavelength/S/N/noise tables without Bokeh.

## Installation

The numerical/text path requires Python 3, NumPy, and Astropy. Install the
package in an environment with:

```bash
python -m pip install -e .
```

Install the optional interactive dependency with
`python -m pip install -e '.[bokeh]'`. The bundled `lrs2_config/` directory
contains the positions, response, and bright/grey/dark sky FITS calibrations;
the package resolves these files relative to itself, not the current directory.

For a Conda environment with the runtime, Bokeh, and test dependencies:

```bash
conda env create -f environment.yml
conda activate lrs2-etc
python -m pip install -e .
```

## Command line

Export one channel without creating HTML:

```bash
lrs2-etc --channel UV --seeing 1.8 --magnitude 19 --sky dark --exptime 1200 \
  --sn-output uv_sn.txt --no-bokeh
```

The table has metadata comments followed by the columns
`wavelength_A  sn_per_pixel  noise`. Export all channels with
`--all-channels --sn-output results/`; this creates one `<channel>_sn.txt`
file per channel. The `--output` option is an alias for `--sn-output`.

By default the command also writes `lrs2_exptime_calc.html`. Choose another
path with `--bokeh-output`, or combine it with `--sn-output` to produce both
outputs. The standalone legacy command remains available as
`python lrs2_exposure_time_calc.py`.
