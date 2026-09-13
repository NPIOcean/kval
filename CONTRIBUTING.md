# Contributing to kval


Any input is valuable - in the form of direct edits to the master branch (for NPIOcean members), 
PRs, and issues/feature requests. 

`kval` is maintained by oceanographers, not software engineers, 
so we lean toward *basic and robust* over *fancy* or *elegant*.

In general, we look toward the [Scientific Python Library Development Guide](https://learn.scientific-python.org/development/) 
when in doubt.

## Built for notebooks

Most `kval` usage happens in Jupyter notebooks, processing a cruise or
mooring dataset interactively rather than as part of a larger pipeline.
Keep this in mind: print feedback and show plots, use sensible defaults - 
help the user along. It's
also why the interactive widget classes (see below) exist.

## Use xarray Datasets

When possible, `kval` uses **xarray Datasets**, which feature intuitive 
organization of data, great support for data and metadata, excellent 
compatibility with NetCDF, and now a large and growing set of other 
scientific libraries built around the class. Other data types including native and numpy 
python types (arrays, dicts etc..) and pandas DataFrames are often used 
within the code itself.

## Don't reinvent the wheel

Where a good, established library already does the job, use it rather
than writing your own version. We lean on `gsw` for TEOS-10 seawater
calculations, `pyrsktools`/RBR's own Ruskin logic for parsing `.rsk`
files, and similar domain libraries elsewhere. That said, this applies
mostly to more complex tasks; if you are implementing a simple function, 
it might be cleaner to write the function than to add an import.  

## Functions, not classes

We try to nudge use in the form:

`ds = function(ds)`

Code structure should default to plain functions: take a `Dataset`, return a new `Dataset`.
Classes are used only for the interactive editing tools (`hand_remove_points`,
`threshold_edit`, etc.). 

## Naming

- An xarray dataset is `ds`
- Internal versions can be `ds_new`, `ds_out`, etc.
- The variable a function acts on is always `variable` (not `var_name`,
`varnm`, `var`).

## Docstrings

Use numpydoc style, e.g. 

```python
def offset(ds: xr.Dataset, variable: str, offset: float) -> xr.Dataset:
    """
    Apply a fixed offset to a specified variable.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset.
    variable : str
        Name of the variable to offset.
    offset : float
        Value to add to the variable.

    Returns
    -------
    xr.Dataset
        A new dataset with the offset applied.
    """
```

It can tedious to write good docstrings in the right format - this is
often a great task to give an AI helper.

## Protect the input

It is often a good idea to start editing functions with:
```python
ds = ds.copy(deep=True)  # Make sure we're not modifying the input ds
```
Callers should not have to worry that calling a `kval` function changed their
original dataset - it can create weird bugs and major headaches. 

(Exception: interactive tools that need to mutate the
same object across clicks — say so explicitly in the docstring if so.)

## `processing_history`

If a function changes a variable's values (editing, filtering,
calibrating, deriving), the changes should be logged in the 
`processing_history` attribute of the variable. 

Use the function `xr_funcs.append_processing_history`, e.g.:

```python
from kval.util import xr_funcs

note = f"Applied a constant offset of {offset:+} {units} to all values."
ds = xr_funcs.append_processing_history(ds, variable, note)
```

- Include actual parameter values and units, not vague descriptions.
- Skip the note entirely if nothing actually happened (a no-op call).

## Tests

`kval` has an extensive pytest library. Ideally we want very good test coverage for 
all functionality - we are far from *full* coverage at the moment, but we are getting 
there. This is *very* useful both in discovering bugs and checking that
new changes don't break anything.

`kval` has *unit tests* of individual functions in `tests/unit_tests`. Each module in 
the source code should have a test script here, e.g. `data/moored.py` has a corresponding 
test script in `tests/unit_tests/data/test_moored.py`.

In addition, we have *functional tests* which are broader tests - like for example a full chain
of processing for a dataset, using different functions from different modules. These are 
found in `tests/functional_tests` - coverage here is relatively thin at the moment.


Pytests are surprisingly easy to write (and "write a pytest for this function" is a great
task for AI), so please consider writing tests accompanying any functions you write. Writing 
a realistic test along with the main code is a great idea as it helps you create a good function.  

If something can only be tested by hand in a notebook (a
widget), it is good to say so in the docstring.

## Using AI tools

We frequently use AI tools in
writing and maintaining `kval`, including for troubleshooting,
sanity-checking logic, writing docstrings, and writing tests. Feel free
to use them - but dont trust them blindly. They make mistakes - sometimes
weird or subtle ones. Good test coverage comes in really handy here as 
it can help steer design and makes it much harder to break anything 

The tool doesn't replace you actively reading and understanding the
change — it's a fast collaborator, not an autopilot.