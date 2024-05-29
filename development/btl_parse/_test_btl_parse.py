from oceanograpy.data import ctd
from oceanograpy.io import cnv

btl_fn = './sta013_01.btl'

### Try loading a header
header = cnv.read_header(btl_fn)

### Try loading the btl data and metadata (one file) to xr Dataset
ds = cnv.read_btl(btl_fn)

### Try loading and concatenating all .btl files from a cruise
path_to_btls = './'
D = ctd.dataset_from_btl_dir(path_to_btls)

### Align with CF-conventions 
# (mostly adds a bunch of global attributes) 

# **Note** This should get us close, but for a typical dataset, we still have to
# add various custom attributes like title, summary, etc.

D = ctd.make_publishing_ready(D)

# Run a convention checker
ctd.check_conventions.check_file(D)

# Export to netCDF