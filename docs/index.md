# dR2star

dR2star is a BIDS-app style wrapper around `tat2` for running on fMRIPrep
derivatives. It scans each `func/` directory, aggregates matching preprocessed
BOLD runs, and writes one output per subject/session.

## Quick start

```sh
dR2star /path/to/fmriprep /path/to/output participant \
  --participant-label 01 02 \
  --ses-label V01 V02 V03
```

See the [Usage](usage.md) page for the full CLI help and options.

The `--concat` option can be used to request concatenation across selected BIDS entities.
