# `oceanograPy`
```
import oceanograpy as opy
```
Collection of Python functions for working with oceanography data processing and analysis.

____
STATUS 18.11:
- Now contains extensive functionality for working with CTD data:
    - Parsing cnvs
    - Reformatting and adding metadata for pulishing CTD data
    - Starting to become quite useful

____

Just starting up, envisioning the following submodules:

- `io`: Converting to and from various file format (e.g. read CTD .cnv data to netCDF) 
- `data`: Data post-processing and QC (e.g. CTD post-processing)
- `plots`: Various tools to help make nice (matplotlib) figures
- `map`: Tools for making maps.
- `geo`: Geographical calculations (coordinate transformations, point-to-point distances etc)  
- `ocean`: Oceanography-specific tools (e.g. vertical modes, turner angles, wkb scaling, geostrophical calculations)
- `calc`: Various useful function
- `util`: Various backend support functions 

Private for now, should make public once there starts to be actual useful stuff here. 
