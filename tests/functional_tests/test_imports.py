import pytest

@pytest.mark.filterwarnings("ignore:The seawater library is deprecated! Please use gsw instead.")
def test_imports():
    from kval.calc import number
    from kval.data import edit, dataset, ctd, moored
    from kval.data.ship_ctd_tools import _ctd_edit, _ctd_tools, _ctd_visualize
    from kval.file import sbe, rbr, matfile,  _variable_defs
    from kval.geo import geocalc
    from kval.maps import quickmap
    from kval.metadata import conventionalize, check_conventions, _standard_attrs, _standard_attrs_org
    from kval.ocean import empirical, uv
    from kval.util import time, user_input, xr_funcs