# About

## Purpose and functionality

- `kval` is a Python library for working with oceanography data processing and analysis.
- It is intended as a rather broad suite of Python tools to work with ocean data and metadata.
    - The scope of `kval` evolves somewhat organically.
- One key goal is to ease the post-processing of oceanographic data (focused on shipboard and moored CTD data) from instrument files to publication-ready NetCDF files.

The library works best in a modern Python environment and using [Jupyter notebooks](https://jupyter.org/).


```{admonition} A warning
:class: warning

*`kval` is in active development - use at your own risk and please be careful!*
```


## Submodules


- `file`: Converting to and from various file format (e.g. read CTD .cnv data to xarray/netCDF)
- `data`: Data post-processing and QC (e.g. CTD post-processing)
- `metadata`: Handling and standardizing metadata according to CF conventions
- `plot`: Plotting tools (e.g. T-S diagrams, maps)
- `geo`: Geographical calculations (coordinate transformations, point-to-point distances etc)
- `ocean`: Oceanography-specific tools (e.g. (`*`) vertical modes, turner angles, wkb scaling, geostrophical calculations)
- `calc`: Various useful functions for numerical calculations.
- `util`: Various backend support functions and wrappers for xarray functionality.
- `signal`: Filtering, spectral analysis, etc.

`*` Not implemented

___

## General principles


*Note: These are aspirational guidelines and not always adhered to in the current code structure. We will try to get there!*

### Development and maintenance

- We will attempt to follow the guidelines from the
  [Scientific Python Library Development Guide](https://learn.scientific-python.org/development/).
- *Releases* will be published relatively often, whenever a new functionality has been added.
   Releases will be archived on [zenodo](https://zenodo.org) and given a DOI.
- See the [contributor guide](https://github.com/NPIOcean/kval/blob/master/CONTRIBUTING.md).


### Code

- Written in Python (>=3.11).
- Tailored for use in a [Jupyter](https://jupyter.org) notebook environment.
- Data and metadata should be stored in [xarray](https://docs.xarray.dev/en/stable/) `Datasets`.
    - Intermediate operations using, e.g., `numpy` or `pandas` objects are fine, but the end user should interact with `Datasets`.
- Visualizations are produced using [matplotlib](https://matplotlib.org/).
- Code should adhere to [PEP8](https://peps.python.org/pep-0008/) style guide, and all functions should have docstrings.
- All functionality should have associated [pytest](https://docs.pytest.org/en/7.4.x/) tests.
    - Unit tests of individual functions are found in `tests/unit_tests/`. Its directory structure and contents should mirror that of `src/kval`.
    - Tests of more complex functionality (e.g. processing pipelines using multiple modules) should be put in  `tests/functional_tests/`.
    - A collection of sample data to be used in testing is found in `tests/test_data/`. Should aim to cover a wide range of input data, but we also don't want this to become *too* bulky - try to keep file size to a minimum.

### Metadata

- All operations that modify data should be recorded in the file metadata.
- Wherever possible, and at as early a stage as feasible, all available useful metadata should be added to Datasets.
- Metadata formatting should adhere to [CF](http://cfconventions.org/) and [ACDD](https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery_1-3) conventions, supplemented by:
    - [OceanSITES](http://www.oceansites.org/docs/oceansites_data_format_reference_manual_20140917.pdf)
    - [2021 NPI NetCDF guidelines](https://gitlab.com/npolar/netcdf-creator/-/blob/main/docs/netcdf_standard_npi.pdf?ref_type=heads)

## People and projects

- The project is maintained by the Oceanography section at the [Norwegian Polar Institute](https://www.npolar.no/en).
- Development is supported by the project [HiAOOS](https://hiaoos.eu/).
- External contributions (pull requests, issues, whatever) are very welcome!


## Latest release


About the latest release, `0.5.0`:


> ***NOTE*** 0.5 introduces **breaking changes** from 0.4. Existing code may need updating. See *Breaking changes* below.

**New**
- `kval.plot.tsplot`: T-S diagram plotting.
- `moored.plot` works on 2D variables.
- Function for combining moored datasets, and time averaging (`xr_funcs.time_average`).
- Other minor support functions.

**Fixes**
- Interactive plots no longer go blank when called more than once in the same notebook. This was caused by a bug in `ipykernel` 7 ([ipykernel#1564](https://github.com/ipython/ipykernel/issues/1564)), so `kval` now requires `ipykernel<7`.
- Interactive hand edits are now applied correctly, and interactive figures display consistently.
- Better drift correction (non-uniform time steps) and time utilities (fractional seconds).
- Various fixes to SBE and RBR file reading, CF-compliance checking (now CF-1.11), and metadata.
- Refactoring and cleaning up of the code.
- Extended pytest coverage.

**Dependencies**
- Temporary caps: `ipykernel<7` and `matplotlib<3.11`. These will be relaxed when the upstream issues are resolved.
- Now requires `compliance-checker>=6.1`; `gsw` and `netcdf4` are listed explicitly.

**Breaking changes**
- Python 3.10 is no longer supported (now 3.11–3.13).
- Renamed modules:
  - `kval.data.ctd` → `kval.data.ctdprof`
  - `kval.maps` → `kval.plot`
  - `kval.metadata.check_conventions` → `kval.metadata.compliance`
- Renamed arguments: `var_name`/`varnm` → `variable`, and `D` → `ds`. Affects code that passes these as keyword arguments.
- The `PROCESSING` variable has been removed. Processing steps are now recorded in a `processing_history` attribute.
- Metadata functions no longer assume NPI-specific attributes.

`kval` is in active development.

## Past releases


About release, `0.4.3`:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17724664.svg)](https://doi.org/10.5281/zenodo.17724664)


- Various fixes, mostly small bugs.
- Removed jupyterlab dependency.
- Resolved some version-related issues leading to problems with interactive plots.
- Cleaned up the source code a bit.


About release, `0.4.2`:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17723685.svg)](https://doi.org/10.5281/zenodo.17723685)


- (issue with mpl version - created 0.4.3 to fix)


About release, `0.4.1`:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17723211.svg)](https://doi.org/10.5281/zenodo.17723211)


- (issue with python version - created new 0.4.2 to fix)


About release, `0.4.0`:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17353994.svg)](https://doi.org/10.5281/zenodo.17353994)


- Adds extended functionality for metadata handling.
- Various other fixes and some added functionality, but not a major overhaul.



About release, `0.3.2`:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15260487.svg)](https://doi.org/10.5281/zenodo.15260487)

- v0.3 used to work out the pypi and conda distribution.
- v0.3.2 adds a basic documentation infrastructure with minimal documentation.
- An updated version with improved documentation and a code overhaul is planned for Spring 2025.