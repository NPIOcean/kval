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

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10360162.svg)](https://doi.org/10.5281/zenodo.10360162)

About the latest release, `0.6.0`:


> ***NOTE*** 0.6 changes some behaviour and defaults. Results can differ slightly from 0.5. 

**New**
- Time averaging function now records what it did in a CF-compatible format (`TIME_bnds`, `cell_methods`), with optional sample counts, standard deviations and medians.
- `combine_datasets` works with calendar intervals; `dataset_size(ds)` shows how large a dataset is.

**Fixes**
- `.cnv` files could lose their first data row; SBE56 `.csv` dates could be read month-first - both fixed
- Fixes to RBR reading, despiking of 2-D data, `principal_angle`, `psd`, magnetic declination and ACDD metadata (`geospatial_bounds`), plus many smaller ones.

**Breaking changes**
- `drop_variables` takes `retain`/`drop` by name only.
- Daily averaging bins run midnight to midnight; `combine_datasets` keeps TIME in its input form.
- float64 is kept; int64 is written as int32.

**Dependencies**
- Requires `pandas>=2.2`; `glob2` and `xlrd` are no longer needed; `kval.util.era5` is removed.

`kval` is in active development.

___