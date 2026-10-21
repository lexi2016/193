# Warmup canary synthetic tests

Before collection, explicitly set both environment variables in your authorized local environment:

- `FACTOR_FACTORY_ROOT`: existing Factor Factory project root containing the bound original formula snapshots/worktree.
- `FACTOR_FACTORY_RAW_READER`: existing, authorized original raw-reader source file.

Use the previously approved training Python environment. No default path, alternate source or producer is selected. Missing variables raise `KeyError` with the variable name.

Targeted path-binding checks, from the implementation worktree:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 python -B -m pytest research/a_share_model_selection_v1/tests/test_environment_tilt_warmup_canary.py -q -p no:cacheprovider -k 'original_raw_reader_on_synthetic_day_bytes_no_second_parser or original_formula_warmup_difference_with_exact_A_reproduction'
```

These checks use synthetic rows only. They do not activate a real canary or change prior execution bindings.
