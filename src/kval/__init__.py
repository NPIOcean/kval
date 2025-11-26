# kval

import matplotlib

def _check_widget_compatibility():
    import warnings
    mver = tuple(map(int, matplotlib.__version__.split('.')[:2]))
    if mver >= (3, 9):
        warnings.warn(
            "Matplotlib >=3.9 may in some cases break ipympl interactive plots. "
            "If you experience trouble with plots not displaying correctly, try downgrading to matplotlib 3.8.x.",
            UserWarning
        )


# Run check automatically when this module is imported
_check_widget_compatibility()