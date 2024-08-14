## Troll transect 22/23 CTD data

A small subset of CTD data from the 2022/2023 Troll Transect cruise.(Malik Arctica?)

Collected using a SBE911+. Data have been preprocessed with SBE software, but not binned.

Including a subset of 3 of 12 profiles. The cnvs are *non-binned* data, which it is useful 
to include in the test suite. However, this makes the full files much larger (>15MB) than I am happy to keep in the repo.

Therefore: cut away most of the profiles to get each files down to about 1MB (still not ideal,
but good enoug for now). Cut away most of the surface soak and then everything after hitting about 100 dbar.