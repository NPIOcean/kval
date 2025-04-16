![image](graphics/kval_banner.png)

Collection of Python functions for working with oceanography data processing and analysis.

Maintained by the Oceanography section at the [Norwegian Polar Institute](https://www.npolar.no/en/).

Supported by the project [HiAOOS](https://hiaoos.eu/).
___

Last release,`0.3.0`:



[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15229098.svg)](https://doi.org/10.5281/zenodo.15229098)



- This version is used to work out pypi and conda distribution.
- An updated version with improved documentation and a code overhaul is planned for Spring 2025.

__

*In active development.*

____
<details>
<summary><strong>CORE FUNCTIONALITY</strong></summary>

##### Submodules


- `file`: Converting to and from various file format (e.g. read CTD .cnv data to xarray/netCDF)
- `data`: Data post-processing and QC (e.g. CTD post-processing)
- `metadata`: Handling and standardizing metadata according to CF conventions
- `plots*`: Various tools to help make nice (matplotlib) figures
- `maps`: Tools for making maps
- `geo`: Geographical calculations (coordinate transformations, point-to-point distances etc)
- `ocean`: Oceanography-specific tools (e.g. (`*`) vertical modes, turner angles, wkb scaling, geostrophical calculations)
- `calc`: Various useful functions for numerical calculations.
- `util`: Various backend support functions and wrappers for xarray functionality.
- `signal`: Filtering, spectral analysis, etc.

`*` Not implemented

</details>

____

<details>
<summary><strong>GENERAL PRINCIPLES</strong></summary>

*Note: These are aspirational guidelines and not always adhered to in the current code structure. We will try to get there!*

###### Code

- Written in Python (>=3.8).
- Tailored for use in a [Jupyter] notebook environment.
- Data and metadata should be stored in [xarray(https://docs.xarray.dev/en/stable/)] `Datasets`.
    - Intermediate operations using, e.g., `numpy` or `pandas` objects are fine, but the end user should only interact with `Datasets`.
- Code should adhere to [PEP8](https://peps.python.org/pep-0008/) style guide, and all functions should have docstrings.
- All functionality should have associated [pytest](https://docs.pytest.org/en/7.4.x/) tests.
    - Unit tests of individual functions are found in `tests/unit_tests/`. Its directory structure and contents should mirror that of `src/kval`.
    - Tests of more complex functionality (e.g. processing pipelines using multiple modules) should be put in  `tests/functional_tests/`.
    - A collection of sample data to be used in testing is found in `tests/test_data/`. Should aim to cover a wide range of input data, but we also don't want this to become *too* bulky - try to keep file size to a minimum.

###### Metadata

- All operations that modify data should be recorded in the file metadata.
- Wherever possibly, and at as early a stage as possible, all available useful metadata should be added to Datasets.
- Metadata formatting should adhere to [CF](http://cfconventions.org/) and [ACDD](https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery_1-3) conventions, supplemented by:
    - [OceanSITES](http://www.oceansites.org/docs/oceansites_data_format_reference_manual_20140917.pdf)
    - [2021 NPI NetCDF guidelines](https://gitlab.com/npolar/netcdf-creator/-/blob/main/docs/netcdf_standard_npi.pdf?ref_type=heads)

###### Project

- The project is maintained by the Oceanography section at the [Norwegian Polar Institute](www.npolar.no/en).
    - External contributions (pull requests, issues, whatever) are very welcome!
- We will attempt to follow the guidelines from the
  [Scientific Python Library Development Guide](https://learn.scientific-python.org/development/).
- *Releases* will be published relatively often, whenever a new functionality has been added.
   Releases will be archived on [zenodo](www.zenodo.org) and given a DOI.

###### Contributing

Pull requests, issues, etc are very welcome!

*[Something about branches here]*

*[Something about python style, test suite]*


</details>

____


<details>
<summary><strong>RELEASE NOTES</strong></summary>





- **0.3.0** *(CURRENT, released 16 April 2025)*:
    - Refactoring of large parts of the code for structure, clarity and efficiency.
    - ~~Complete reproducability and self-documentation in the CTD processing functionality.~~
    - Adding mooring processing functionality
        - ~~Functionality for parsing moored CTD sensors (RBR and SBE)~~
        - Functionality for basic processing source file -> CF-NetCDF
    - ~~Test suite (comprehensive but not entirely complete)~~
    - ~~Decluttering the repo bringing it down from its currently bloated state.~~
    - Started useful documentation with somenotebook examples (need a lot of work).
    - Released to PyPi and conda-forge.
    - Many other improvements and additions.


- **0.4.0** *(PLANNED for spring 2025)*

- Complete (useful) documentation with good notebook examples of core functionality.
- Expanded test suite for good coverage of basic functionality plus
- *Possibly*:
    - Parsing data from inductive loggers.
    - Breaking out NPI-specific stuff (if there really is that much of it) to a sepåarate libraru, ´naust´.


- *0.0.2*:
    - Name change from `oceanograpy` to `kval`.
    - Introduction of test suite.
    - Other minor changes.

- *0.0.1:*
    - Initial release.
    - Functionality tailored for CTD processing.




</details>




