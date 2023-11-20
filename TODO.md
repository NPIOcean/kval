# TO DO


### ØL 20.11

Added functionality for parsing .btl files. Seems to work well, but the structure may need some work:

- Some code is doubled up between cnv and btl stuff in `_ctd_tools` (see `join_cruise_btl`)
- Some function/module names may now not be fitting anymore - should go through and look at names.
  - I.e., the `cnv` module is now used to parse `.btl` and probably `.hdr` files.  

These files are probably a good starting point fo rthe salinometer comparison. May require
an `export_to_exel` or `import_from_excel` function or something. This should line up with 
standardization of sample and salinometer stuff which I will have to do soon.


Also:

  - Add **dependencies** in `pyproject.toml`! Want other people to be able to run the 
    development notebooks after install.
  - Also work on these notebooks a bit..


### ØL 17.11

- CTD parsing -> joining -> metadata formatting now works pretty well after a bit of a marathon.
    + Still want to look at th structure and see if we should refactor something.
    + Probably want to make a nicer way of filling in custom metadata. Currently using a dict in an
      adjacent .py file; this works well but can be simplified/standardized.

- Next up is building CTD processing tools. Things should be set up well for working on this now.
  Want to use widgets where possible, but need to make sure things are robust and test extensively
  across systems and datasets. Proposed functionality:

    + Hand outlier editing.
    + Threshold editing 
    + Runninng filter editing 
    + Applying offsets
    + Anything else?

- All of these except the first should be straight forward to code up. The main challenge is to
  preserve metadata in a sensible way.

    - I have considered making an `INFO` metadata varible in the (pre-published) Dataset. This would
      not contain data, but it would be a good place to keep metadata attributes that are useful
      along the process but that we don't want in the final file. Could presumably be a clean way to
      avoid cluttering up the dataset.
      + May paticularly be useful to kleet track of editing etc. Can be accessed after to create
        summaries of the processing steps 

- Next after that will be to standardize the salinometer bottle file comparisons. 

    + I really think there needs to be two separate files at the end, so I don't think this has to
      bee too ingrained in the ctd profile functionality.
    + The header parsing might work well from .btl data, though, and it might be nice to extract the
      info the same way for btl and cnv in order to get i.e. the same TIME coordinate and metadata. 

- Also want to build some basic analysis tools. Some of them may overlap with the QC tools:

    - Profile, TS, and contour plots.
    - Histograms.
    - Map plots.
    - Line plots of selected variables at selected depths (function of time).

- I think there is somethign to be said for building the analysis tools *before* the QD/editing
  tools so we can use the formed in the latter.

When all of this, or at least the CTD part, is done, it might be good to create a **version**.
Should have some decent examples and documentation to go along with it. This is probably good
practice, and I assume it will make it easier to control the package in prescribed conda envs, for
example.

- Maybe try the sphinx approach since the docstrings are generally decent? 
- Seems like this should live on *HavData*/*HavNett*

Eventually, I should work on good integration with matlab. At least parsing back and forth should be
made very easy. If we need to have the option of editing in matlab: should think of a way to
    preserve history. - (E.g. keep the unedited file and compare it with the edited one to deduce
    what happened and retain metadata?)

### Generally about oceanograPy

- Define some *code guidelines*! 
- Define *contributor guidelines:* e.g.maybe ØL/TH or anyone else with some overview should review
  PRs for this one..
- Think about *tests*!
    + Try smple, basic tests to begin with -> Need to get used to how this works. 
- Move things over from the old `oyv` module - but be a little discriminatory, want to reserve this
  for relatively useful stuff and relatively clean code.
- For all of this the [ Scientific Python Library Development Guide
](https://learn.scientific-python.org/development/) shod be a good guide.
