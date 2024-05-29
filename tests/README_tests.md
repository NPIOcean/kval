# About the tests

Will aim to maintain two types of tests:

1. **Unit tests** testing individual functions.

    - Located in `tests/unit_tests`. 
    - The contents of `unit_tests` should mirror `src/kval`. I.e., there should be a test script `tests/unit_tests/calc/test_numbers.py` 
      containing tests of the functions in `src/kval/calc/numbers.py`. 

2. **Functional tests** testing more complex operation spanning the `kval` library.

    - Located in `tests/functional_tests`. 
    - Should have tests of loading data from source files and applying processing steps, etc.
    - Souce data for these tests should be located in `tests/functional_tests/test_data/`.


Should have at least one unit test of each function - good practice to make the test when writing a function (useful in development as well!).

Want to have comprehensive functional tests that can help make sure small changes don't break the core functionality.

Want to test different input formats. 

Eventually want to implement automatic testing upon commits or something (look into later).