[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18929595.svg)](https://doi.org/10.5281/zenodo.18929595)
# dR2*

dR2star is a BIDS-App style workflow for deriving dR2*/R2*-related estimates
from single-echo fMRI data that have already been preprocessed with fMRIPrep.
It discovers matching preprocessed BOLD files, confounds TSVs, and reference
masks, then runs the dR2* calculations on the selected volumes.

## Quick Start

The recommended way to run dR2star is via the public container image hosted on
GitHub Container Registry:

- Package page: `https://github.com/Larsen-Lab/dR2star/pkgs/container/dr2star`
- Container name: `ghcr.io/larsen-lab/dr2star`

Pull with Apptainer/Singularity:

```sh
apptainer pull dR2star.sif docker://ghcr.io/larsen-lab/dr2star:latest
```

Run with Apptainer/Singularity:

```sh
apptainer run --cleanenv \
  -B /path/to/fmriprep:/input_dir \
  -B /path/to/output:/output_dir \
  dR2star.sif \
  /input_dir /output_dir participant
```

Pull with Docker:

```sh
docker pull ghcr.io/larsen-lab/dr2star:latest
```

Run with Docker:

```sh
docker run --rm \
  -v /path/to/fmriprep:/input_dir \
  -v /path/to/output:/output_dir \
  ghcr.io/larsen-lab/dr2star:latest \
  /input_dir /output_dir participant
```

Use a release tag instead of `latest` when you want a fixed container version.

## Required Inputs

dR2star expects an fMRIPrep derivatives directory containing, for each run you
want to analyze:

- a confounds TSV named like
  `*_desc-confounds_timeseries.tsv` or `*_desc-confounds_regressors.tsv`
- the matching preprocessed BOLD file named like
  `*_space-<space>_desc-preproc_bold.nii.gz`
- a matching fMRIPrep brain mask in the same space, or a custom mask supplied
  through `--reference-mask-input`

The main input should already be organized as BIDS derivatives under
`sub-*/[ses-*/]func/`.

## Outputs

Outputs are written under:

```text
OUTPUT_DIR/sub-<label>/[ses-<label>/]anat/
```

Each processed run or concatenated group produces:

- a dR2*/R2*-related map:
  `*_desc-dR2star_dR2starmap.nii.gz`
- a JSON sidecar describing provenance, volume selection, and mask handling
- optional intermediate files when requested

## Full Documentation

For complete usage instructions, container details, expected outputs, and
maintainer guidance, use the Read the Docs site:

- User documentation: `https://dr2star.readthedocs.io/en/stable/`
- Maintainer documentation:
  `https://dr2star.readthedocs.io/en/stable/maintainers.html`

The GitHub README is intentionally brief. Read the Docs should be the primary
reference for day-to-day use and maintenance.

## Provenance

Extracted from [lncdtools](https://github.com/lncd/lncdtools) on 2026-01-08.
