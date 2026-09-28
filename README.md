![image](https://raw.githubusercontent.com/npiocean/kval/master/graphics/kval_banner.png)

Collection of Python tools for working with oceanography data processing and analysis.

Maintained by the Oceanography section at the [Norwegian Polar Institute](https://www.npolar.no/en/) and supported by the [HiAOOS](https://hiaoos.eu/) project.
___


#### [Documentation page](https://kval.readthedocs.io/) *(in development)*


___

### Installation

Conda:

    conda install -c npiocean -c conda-forge kval


Pip [^tag] :

    pip install kval


[^tag]: Conda is recommended for Windows users as we have experienced errors with the dependent library `compliance-checker` on Windows+pip.


___

About the latest release, `0.5.0`:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17724664.svg)](https://doi.org/10.5281/zenodo.17724664)

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

___