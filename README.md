![image](graphics/kval_banner.png)

Collection of Python functions for working with oceanography data processing and analysis.

Maintained by the Oceanography section at the [Norwegian Polar Institute](https://www.npolar.no/en/). 
___

Last release,`0.0.2-beta`:

[![DOI](https://zenodo.org/badge/711439231.svg)](https://zenodo.org/doi/10.5281/zenodo.10360162)

___

*Note:* This library was until recently called `oceanograpy`

*In active development.*

____
<details>
<summary><strong>CORE FUNCTIONALITY</strong></summary>

##### Submodules

- `io`: Converting to and from various file format (e.g. read CTD .cnv data to netCDF) 
- `data`: Data post-processing and QC (e.g. CTD post-processing)
- `plots`: Various tools to help make nice (matplotlib) figures
- `map`: Tools for making maps.
- `geo`: Geographical calculations (coordinate transformations, point-to-point distances etc)  
- `ocean`: Oceanography-specific tools (e.g. vertical modes, turner angles, wkb scaling, geostrophical calculations)
- `calc`: Various useful function
- `util`: Various backend support functions 
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
    - A collection of sample data to be used in testing is found in `tests/test_data/`. Should aim to cover a wide range of input data, but we also don't want this to become *too* bulky.

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

</details>

____


<details>
<summary><strong>RELEASE NOTES</strong></summary>

- *0.3.0 (in development)*: 
    - Packaging and hopefully release to PyPi and conda-forge.
    - Removal of NPI-specific content.
    - Work on ensuring complete reproducability of the CTD processing functionality.
 
- *0.2.0*: 
    - Name change from `oceanograpy` to `kval`.
    - Introduction of test suite.
    - Other minor changes.

- *0.1.0:* 
    - Initial release.
    - Functionality tailored for CTD processing.

</details>



____


<details>
<summary><strong>STATUS UPDATES</strong></summary>

____

STATUS 21.04.24:

Renamed from `oceanograpy` to `kval`

STATUS 08.12.23:

- Developed core functionality for editing CTD data. Relies pretty heavily on Jupyter/interactive widgets. 

*TO DO:*

- A look-over of the entire codebase with the view of cleaning up the structure.
    - There is a bit of an unholy mixture between general and specific functions. Should
      make specific modules that are either general or application specific, and give
      them names and locations that reflect their use.
- Test suite (!)
- More extensive documentation/example scripts.

____

STATUS 18.11.23:
- Now contains extensive functionality for working with CTD data:
    - Parsing cnvs
    - Reformatting and adding metadata for pulishing CTD data
    - Starting to become quite useful

____

</details>


