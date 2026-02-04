Container Usage
---------------
This page covers how to find, download, and run the dR2star container using
Apptainer. It is intended for novice users and focuses on the most common
workflow.

Find the latest container
-------------------------
Container images are published via GitHub Packages. The list of available tags
and versions is here:

.. code-block:: text

   https://github.com/Larsen-Lab/dR2star/pkgs/container/dr2star

Download with Apptainer
-----------------------
Pick a tag from the package page and pull it with Apptainer. The example below
uses a placeholder tag. Replace ``<TAG>`` with a real version string.

.. code-block:: sh

   apptainer pull dr2star.sif docker://ghcr.io/larsen-lab/dr2star:<TAG>

If you already have a ``.sif`` file, you can skip this step.

Run the container
-----------------
You must bind your input and output directories into the container and then
pass the in-container paths to the command.

.. code-block:: sh

   input_dir=/path/to/fmriprep
   output_dir=/path/to/output

   apptainer run \
     -B ${input_dir}:/input_dir \
     -B ${output_dir}:/output_dir \
     dr2star.sif \
     /input_dir /output_dir participant \
     --participant-label 001

Binding basics
--------------
- The left side of ``-B`` is a host path; the right side is the path inside the
  container.
- Bind all inputs you want the container to read (fMRIPrep outputs, custom
  masks, confounds).
- Bind the output directory so results are saved outside the container.
- Add extra ``-B`` entries as needed for additional inputs.

If you want to see all available CLI options, run:

.. code-block:: sh

   apptainer run dr2star.sif --help
