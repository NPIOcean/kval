import pytest

@pytest.mark.filterwarnings("ignore:The seawater library is deprecated! Please use gsw instead.")
def test_imports():
    from kval.calc import number
    from kval.data import edit, dataset, ctdprof, moored
    from kval.data.ctdprof_tools import _ctdprof_edit, _ctdprof_tools, _ctdprof_visualize
    from kval.data.moored_tools import _moored_tools, _compare_moored_prof
    from kval.file import sbe, rbr, matfile,  _variable_defs
    from kval.geo import geocalc
    from kval.plot import quickmap, maptools, tsplot
    from kval.metadata import conventionalize, compliance, io, _standard_attrs, _standard_attrs_org
    from kval.ocean import empirical, uv
    from kval.signal import despike, filt, interp, spec
    from kval.util import time, user_input, xr_funcs, index, internals, magdec, netcdf