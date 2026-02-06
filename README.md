# dR2*
## Usage
This repo provides a BIDS-app style wrapper (`dR2star`) around `dr2`
pipeline for running on fMRIPrep derivatives. The wrapper scans each
`func/` directory, aggregates matching preprocessed BOLD runs, and
writes one output per subject/session.

`dR2star` is the wrapper CLI and `dr2` is the underlying processing binary.

High-level flow:
 - point to an fMRIPrep derivatives folder
 - choose an output folder
 - optionally limit participants/sessions

Example:
```
dR2star /path/to/fmriprep /path/to/output participant \
  --participant-label 01 02 \
  --ses-label V01 V02 V03
```

## Output
Outputs are written under:
```
OUTPUT_DIR/sub-<label>/ses-<label>/anat/
```

Each session produces:
 - `sub-*_ses-*_space-MNI152NLin6Asym_res-2_desc-dR2star_dR2starmap.nii.gz`
 - a JSON sidecar with the same basename
 - one or more censor files with matching basename

## Container Details For Whoever is Maintaining This Repo
There are two github actions mechanisms that will result in a new
Docker container being built. First, any changes to the "main" branch
of the repository will result in a new version of the container with the
tag "latest". At the time of writing, it takes less than 5 minutes for a
new version of the container to appear following code updates.

The second mechanism is through a tagged release. To create a new tagged
version of the container, publish a GitHub Release. The workflow will
use the release tag as the container tag, stripping a leading "v" if
present (for example, "v1.2.3" becomes "1.2.3"). The container is
pushed to GitHub Container Registry (GHCR) at:

  ghcr.io/larsen-lab/dr2star

In summary:
  - push to main -> updates ghcr.io/larsen-lab/dr2star:latest
  - publish a release -> publishes ghcr.io/larsen-lab/dr2star:<tag>

If you need to change the branch used for latest or add multi-arch
builds, edit `.github/workflows/docker-publish.yml`.

## Container Details For End Users
The container image is published to GitHub Container Registry (GHCR)
as `ghcr.io/larsen-lab/dr2star`.

Docker pull:
```
docker pull ghcr.io/larsen-lab/dr2star:latest
```

Docker run:
```
docker run --rm \
  -v /path/to/fmriprep:/input_dir \
  -v /path/to/output:/output_dir \
  ghcr.io/larsen-lab/dr2star:latest \
  /input_dir /output_dir participant
```

Singularity/Apptainer pull:
```
apptainer pull dR2star.sif docker://ghcr.io/larsen-lab/dr2star:latest
```

Singularity/Apptainer run:
```
apptainer run --cleanenv \
  -B /path/to/fmriprep:/input_dir \
  -B /path/to/output:/output_dir \
  dR2star.sif \
  /input_dir /output_dir participant
```


## Developing
See tests in `t/`. Run with `make check`

## Provenance
Extracted from [lncdtools](https://github.com/lncd/lncdtools) on 2026-01-08.
```
git clone --branch dr2star-fmriprep --single-branch lncdtools dR2star
find -iname '*dR2star*' -not -ipath '*.git/*' |
  sed 's:^./:--path :'|
  xargs uv tool run git-filter-repo --force \
    --path Makefile \
    --path .github/workflows/ci.yml \
```
