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

> ***NOTE*** 0.6 changes some behaviour and defaults. Results can differ slightly from 0.5. 

___

About the latest release, `0.6.1`:

**Changes**
- Consistent behaviour: Functions in moored, ctdprof and metadata no longer change the input dataset. Assign the result: `ds = func(ds, ...)`.
- Interactive tools (`moored.plot`, `threshold_pick`, `tsplot_pick`, hand editing, the IOOS compliance report) now keep a static picture or text in the saved notebook, so the output is visible outside a live Jupyter session (e.g. on GitHub). This is not necessarily perfect (the best way to view notebooks is in JupyterLab), but it is better than displaying an empty or broken plot.
    - Tested with both ipywidgets 7 and 8.

**Fixes**
- `threshold_pick`: "Apply" now removes exactly what the preview shows; limits that cut nothing no longer add attributes.
- `ctdprof.hand_remove_points` returned the wrong object.
- `edit.replace` failed on datasets with scalar coordinates (e.g. LATITUDE/LONGITUDE).
- `to_netcdf` gives a clear error when the output directory does not exist (previously reported as "file already exists").

___


`kval` is in active development.

___