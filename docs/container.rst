Container Details
=================

This page covers how to find, download, and run the dR2star container with
either Apptainer/Singularity or Docker.

Find the public container
-------------------------

Container images are published through GitHub Container Registry (GHCR).

- Package page:
  ``https://github.com/Larsen-Lab/dR2star/pkgs/container/dr2star``
- Container name:
  ``ghcr.io/larsen-lab/dr2star``

Use ``latest`` for the current ``main`` branch build, or choose a specific
release tag when you want a fixed version.

Download with Apptainer/Singularity
-----------------------------------

.. code-block:: sh

   apptainer pull dR2star.sif docker://ghcr.io/larsen-lab/dr2star:latest

To use a release-specific container instead of ``latest``:

.. code-block:: sh

   apptainer pull dR2star.sif docker://ghcr.io/larsen-lab/dr2star:<TAG>

If you already have a ``.sif`` file, you can skip this pull step.

Run with Apptainer/Singularity
------------------------------

You must bind your input and output directories into the container and then
pass the in-container paths to the command.

.. code-block:: sh

   input_dir=/path/to/fmriprep
   output_dir=/path/to/output

   apptainer run --cleanenv \
     -B ${input_dir}:/input_dir \
     -B ${output_dir}:/output_dir \
     dR2star.sif \
     /input_dir /output_dir participant \
     --participant-label 001

Download with Docker
--------------------

.. code-block:: sh

   docker pull ghcr.io/larsen-lab/dr2star:latest

To use a release-specific container instead of ``latest``:

.. code-block:: sh

   docker pull ghcr.io/larsen-lab/dr2star:<TAG>

Run with Docker
---------------

Bind your fMRIPrep derivatives and output directory into the container, then
pass the in-container paths to ``dR2star``.

.. code-block:: sh

   input_dir=/path/to/fmriprep
   output_dir=/path/to/output

   docker run --rm \
     -v ${input_dir}:/input_dir \
     -v ${output_dir}:/output_dir \
     ghcr.io/larsen-lab/dr2star:latest \
     /input_dir /output_dir participant \
     --participant-label 001

Binding basics
--------------

- The left side of ``-v`` or ``-B`` is a host path; the right side is the path
  visible inside the container.
- Bind all inputs you want the container to read, including fMRIPrep outputs
  and any custom reference masks.
- Bind the output directory so results are written outside the container.
- Add extra binds as needed for additional inputs.

If you want to see all available CLI options, run:

.. code-block:: sh

   apptainer run dR2star.sif --help

or

.. code-block:: sh

   docker run --rm ghcr.io/larsen-lab/dr2star:latest --help
