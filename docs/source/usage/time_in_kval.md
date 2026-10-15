# Time in kval (draft)

How kval represents, computes and describes time, from reading instrument
files to the metadata in the exported netCDF file.

## Time zone: UTC throughout

kval treats every time as UTC. Times are stored without a time zone:

- as `datetime64` (no time zone), or
- as CF numeric time, e.g. `days since 1970-01-01`, with no offset in the
  units. CF reads this as UTC.

ISO 8601 strings written by kval end in `Z`, e.g. `2024-03-01T00:00:00Z`.

kval cannot tell whether an instrument clock was set to UTC; it takes the
times in the files as they are. For SBE `.cnv` files the times are computed
from the start time in the header: the `NMEA UTC` line (UTC from the ship's
GPS) when `start_time_NMEA=True`, otherwise the `start_time` line (the
instrument/deck unit clock).

## Two representations of TIME

| | numeric (CF-encoded) | datetime64 |
|---|---|---|
| Example | `19800.5`, units `days since 1970-01-01` | `2024-03-18T12:00` |
| Where | what the readers (`load_moored`, `read_cnv`, `read_rsk`) return, and what is written to file | after `xr.open_dataset` with default decoding, or `time_as_datetime` |
| Needs | `units` (and `calendar`) attributes | nothing |

Convert a dataset between the two with
`kval.util.xr_funcs.time_as_datetime(ds)` and `time_as_float(ds)`.
`time_as_float` re-uses the units and calendar remembered by
`time_as_datetime`, or takes them as arguments, or defaults to
`days since 1970-01-01 00:00` with a warning.

Functions that need datetimes internally (e.g. `time_average`,
`combine_datasets`) decode TIME themselves and return it in the form it came
in.

The standard TIME attributes kval adds are `units: days since 1970-01-01`,
`calendar: standard`, `axis: T` and `units_metadata: leap_seconds: unknown`
(CF 1.11).

## Reading instrument files

- **SBE `.cnv`**: TIME is computed from the header start time and the
  elapsed-time column (e.g. `timeJ`, `timeS`). Floating-point noise from the
  conversion is removed: records sampled at whole-second intervals are
  rounded to whole seconds, everything else to the millisecond (and never
  so that two samples get the same time).
- **SBE56 `.csv`**: the order of year, month and day in the dates is
  decided from all dates in the file (or given with `date_format=`), not
  guessed by pandas from the first row.
- **RBR `.rsk`**: TIME comes from the sample timestamps in the file.

All give numeric TIME in `days since 1970-01-01`.

## Time averaging (`time_average`)

`time_average(ds, interval)` averages all numeric TIME-dependent variables
in bins of `interval`.

**Bins.** Each bin is `[start, end)`: a sample exactly on the boundary goes
in the *later* bin. Fixed-length intervals (`'1D'`, `'6h'`, `'30min'`, ...)
are anchored to 1970-01-01 00:00 (`origin='epoch'`), so daily bins run
midnight to midnight, 6-hourly bins start at 00/06/12/18, and two datasets
averaged separately share the same bin edges. Calendar intervals
(`'MS'`/`'ME'` months, `'YS'`/`'YE'` years, `'W'` Monday-Sunday weeks) follow
the calendar.

**Timestamps.** `label='center'` (default) puts each timestamp at the
middle of its bin; `'left'` and `'right'` at the start and end. For calendar
months the centres are not evenly spaced.

**What is recorded in the output (CF conventions):**

- `TIME_bnds(TIME, bnds)`: the start and end of each bin, and
  `TIME:bounds = "TIME_bnds"` (CF 7.1). The bounds have no attributes of
  their own; they use those of TIME.
- `cell_methods = "TIME: mean"` on each averaged variable (CF 7.3).
- `TIME:time_averaging`: a sentence saying what the timestamps mean, and
  `TIME:long_name`, e.g. `Time (centre of averaging bin)`.
- The global `time_coverage_resolution` of the input is removed (it
  described the input sampling).

**Incomplete bins.** By default all bins are kept, with a warning if the
first or last bin is less than 95 % full. `min_fraction=0.9` drops bins
with less than 90 % of the expected samples (fixed-length intervals only).

**Extra statistics (opt-in).** `stats=['n_samples', 'std', 'median']` adds,
for each variable (here TEMP):

- `TEMP_n_samples`: number of non-NaN samples per bin
  (`standard_name: number_of_observations`), listed in
  `TEMP:ancillary_variables`;
- `TEMP_std`: sample standard deviation (ddof=1),
  `cell_methods: TIME: standard_deviation`;
- `TEMP_median`: `cell_methods: TIME: median`.

## Combining instruments (`combine_datasets`)

`combine_datasets(ds1, ds2, ..., interval=...)` puts several datasets on one
common TIME grid and stacks them along `INSTR`.

- `method='time_average'` (default): each dataset is averaged with
  `time_average` (`label='left'`), so the grid times are bin **starts**, and
  `TIME_bnds` holds the bins.
- `method='interpolate'`: values are interpolated to the grid times; no
  bounds (the values are instantaneous).

The grid is anchored to the epoch for fixed intervals, and to calendar
periods for calendar intervals. TIME is returned numeric if all inputs had
numeric TIME, otherwise as datetime64.

## Time metadata (`add_range_attrs`, ACDD 1.3)

- `time_coverage_start` / `time_coverage_end`: first and last time stamp
  (what ACDD describes and the IOOS checker tests). With
  `time_coverage_from_bounds=True` and averaged data: start of the first
  and end of the last bin instead.
- `time_coverage_duration`: end minus start.
- `time_coverage_resolution`: the time step if it is constant, otherwise
  `variable`.

Durations use the ISO 8601 format ACDD recommends,
`P[YYYY]-[MM]-[DD]T[hh]:[mm]:[ss]`, e.g. `P0000-00-00T00:15:00`. More than
30 days without months (e.g. a 150-day sampling interval) does not fit
that format, so it is written as `P150D` (ISO 8601 designator format, also
accepted by ACDD).

## Open questions / notes to self

- Leap seconds: `units_metadata = "leap_seconds: unknown"`. Instrument
  clocks probably do not count leap seconds; not looked into.
- Instrument clock drift is not handled anywhere.
