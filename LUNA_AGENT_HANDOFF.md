# Luna handoff: make `lrs2_etc` installable and export S/N tables

## Objective

Turn this repository into a small, version-controlled, installable LRS2 exposure-time calculator that:

1. preserves the interactive Bokeh calculator;
2. writes wavelength/S/N/noise text tables for one channel or all channels;
3. works on a fresh Python 3 installation on macOS and Linux;
4. documents and packages the required calibration data.

## Initial contents

- `lrs2_exposure_time_calc.py` is the legacy calculator copied from Panacea.
- `lrs2_config/` contains the required positions, response, and bright/grey/dark sky FITS files.
- The source calculator currently loads files relative to the current working directory and builds the Bokeh application at import time.

The linked `dichroic_uv_orange_fit.py` from the parent Panacea repository is a separate workflow and is not required for this exposure-time calculator. Do not add it unless there is a clear reason.
please investigate what this sentence above means as well, why is dichroic_uv_orange_fit.py linked, we don't need it

## Required implementation

### Refactor the calculator

- Separate calibration loading, numerical S/N calculation, text export, and Bokeh UI construction.
- Keep a reusable numerical function returning `(wavelength, sn_per_pixel, noise)`.
- Add an `argparse` CLI with options for channel, seeing, AB magnitude, sky kind, exposure time, output path, and an option to process all channels.
- Write plain text/ASCII tables with a metadata header and columns:

  ```text
  wavelength_A  sn_per_pixel  noise
  ```

- Support producing text files alone, Bokeh alone, or both in one invocation.
- Use `pathlib` and resolve the default `lrs2_config` relative to the package/repository rather than the caller's working directory.
- Use the actual lowercase calibration filenames so the code works on case-sensitive filesystems.
- Close FITS files with context managers.
- Remove the unused pandas dependency and the erroneous `DF = pd.DataFrame = response_dict` assignment.
- Update the Python 2 shebang/documentation to Python 3.

### Validate the numerical model

- Reconcile the Python and JavaScript noise calculations. They currently differ in the final weighting: Python uses a factor of `weights`, while JavaScript effectively uses `weights**2`.
- Ensure the initial Bokeh plot displays the default S/N curve instead of an all-zero curve.
- Add tests covering shape, finite output, channel/sky validation, exposure-time scaling, and agreement between the text and Bokeh numerical paths.

### Packaging and documentation

Create:

- `pyproject.toml` with a Python 3 requirement, runtime dependencies, package data, and console entry point(s).
- `environment.yml` for a reproducible Conda environment.
- A minimal `README.md` explaining installation, calibration data, CLI examples, output format, and Bokeh usage.
- Appropriate package/module layout if needed, while keeping the FITS calibration files included in distributions.

The text-only path should require NumPy and Astropy. Bokeh should be an optional dependency if practical; pandas is not needed. Use the repository's existing style and add focused tests rather than copying the full Panacea package.

## Suggested acceptance checks

```bash
python -m pip install -e .
python -m pytest
lrs2-etc --channel UV --seeing 1.8 --magnitude 19 --sky dark --exptime 1200 \
  --sn-output uv_sn.txt --no-bokeh
```

The command should create a readable table with one row per wavelength sample. A separate command or flags should create the Bokeh HTML file, and a combined invocation should be able to create both outputs.

Do not modify the parent Panacea repository or its calibration files while implementing this task.
