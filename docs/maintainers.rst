Information for Maintainers
===========================

This page is for people maintaining the repository, documentation, container,
and release process.

Repository layout
-----------------

The codebase is split between a Python orchestration layer and the lower-level
``dr2`` shell script that implements the AFNI-backed image processing.

- ``dR2star/my_parser.py`` defines the public CLI, argument help text, and the
  options rendered into the user documentation through ``sphinx-argparse``.
- ``dR2star/run.py`` is the main workflow entry point. It discovers subjects,
  sessions, confounds files, matching preprocessed BOLD files, and masks; it
  groups runs when ``--concat`` is used; it applies confound-based volume
  selection; and it calls ``dr2`` through ``subprocess.run``.
- ``dR2star/utilities.py`` holds most of the reusable workflow helpers, such as
  session and subject discovery, mask lookup, mask resampling, volume selection
  from confounds, and JSON post-processing.
- ``dr2`` is the lower-level executable that performs the AFNI-based processing
  and writes the base provenance JSON consumed by the Python wrapper.
- ``Dockerfile`` defines the container image used for Docker and Apptainer/
  Singularity execution.
- ``.github/workflows/docker-publish.yml`` builds and publishes the GHCR
  container image.
- ``docs/`` contains the Sphinx documentation source rendered on Read the Docs.

At a high level, the Python code is responsible for BIDS/fMRIPrep-specific
input discovery, bookkeeping, metadata handling, and workflow control. The
``dr2`` script is responsible for the underlying voxelwise computation.

Changing the documentation
--------------------------

The Read the Docs site is built from the files in ``docs/``. The main entry
points are:

- ``docs/index.rst`` for the landing page and navigation
- ``docs/usage.rst`` for CLI behavior and workflow details
- ``docs/container.rst`` for container usage
- ``docs/expected_outputs.rst`` for output naming and JSON metadata
- ``docs/maintainers.rst`` for maintainer-facing information

The docs configuration is controlled by:

- ``.readthedocs.yaml``: tells Read the Docs which Python version to use and
  which Sphinx configuration file to build
- ``docs/conf.py``: Sphinx settings
- ``docs/requirements.txt``: Python packages needed to build the docs

To preview the docs locally, install the doc requirements and build the HTML:

.. code-block:: sh

   python3 -m pip install -r docs/requirements.txt
   python3 -m sphinx -b html docs docs/_build/html

How the docs reach Read the Docs
--------------------------------

Read the Docs reads ``.readthedocs.yaml``, installs ``docs/requirements.txt``,
and builds the site using ``docs/conf.py``. Once a branch or tag exists on
GitHub, it can be built as a documentation version in the Read the Docs project
dashboard.

In practice:

1. Edit the ``docs/*.rst`` files in this repository.
2. Push the changes to GitHub.
3. Ensure the relevant branch or tag version is enabled on Read the Docs.
4. Trigger or wait for the Read the Docs build for that version.

Version activation, aliases such as ``stable`` and ``latest``, and which branch
or tag is exposed publicly are managed in the Read the Docs project settings,
not in this repository.

Releases, tags, and container images
------------------------------------

The container publishing workflow is defined in
``.github/workflows/docker-publish.yml``.

It runs when:

- code is pushed to ``main``
- code is pushed to ``development``
- a GitHub Release is published

Container images are pushed to:

- ``ghcr.io/larsen-lab/dr2star:latest`` for pushes to ``main``
- ``ghcr.io/larsen-lab/dr2star:development`` for pushes to ``development``
- ``ghcr.io/larsen-lab/dr2star:predevelopment`` for pushes to
  ``predevelopment``
- ``ghcr.io/larsen-lab/dr2star:<tag>`` for published releases

For release tags, the workflow strips a leading ``v`` before assigning the
container tag. For example, a GitHub release named ``v1.2.3`` becomes the
container tag ``1.2.3``.

Recommended release naming
--------------------------

Use semantic version tags such as:

- ``v0.3.0``
- ``v1.0.0``

That keeps GitHub releases readable while still producing clean GHCR image tags
such as ``0.3.0`` and ``1.0.0``.

Managing Read the Docs versions alongside releases
--------------------------------------------------

The public user-facing docs should generally point to the stable release docs,
while branch docs can be used for previews.

A practical workflow is:

1. Use ``development`` and ``predevelopment`` for work-in-progress docs and code.
2. Merge release-ready changes into ``main``.
3. Publish a GitHub release with a semantic version tag such as ``v0.3.0``.
4. In Read the Docs, activate the new tag version if needed and keep the
   ``stable`` alias pointed at the current released documentation.
5. Keep README links pointed at the ``stable`` docs URL for end users.

Current public docs entry point:

- ``https://dr2star.readthedocs.io/en/stable/``

Current public container package page:

- ``https://github.com/Larsen-Lab/dR2star/pkgs/container/dr2star``
