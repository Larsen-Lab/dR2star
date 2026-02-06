Usage
=====
.. argparse::
   :ref: dR2star.my_parser.get_parser
   :prog: dR2star
   :nodefaultconst:

Method Choices
--------------
The ``--dr2star-method`` option accepts the following values (case-insensitive):

- ``dR2star``: log transform of the normalized signal (natural log).
- ``nT2star``: normalized signal proportion without log/z-score transforms.
- ``zscore``: z-score transform of the normalized signal.

Reference Mask Scenarios
------------------------
Use the scenarios below to decide how to set ``--reference-mask-input``. Each scenario shows
both the expected directory layout and a full container command.

**Scenario 1: No custom reference masks (use fMRIPrep-derived brain masks)**

You do not have any custom masks and want dR2star to use fMRIPrep brain masks as the
reference region. Omit ``--reference-mask-input`` and ensure your fMRIPrep ``func/``
directory contains the matching ``*_desc-brain_mask.nii.gz`` files for the requested
``--space``.

Example input layout:

.. code-block:: text

   /path/to/fmriprep/
     sub-001/
       ses-01/
         func/
           sub-001_ses-01_task-rest_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz
           sub-001_ses-01_task-rest_desc-confounds_timeseries.tsv
           sub-001_ses-01_task-rest_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz

Example command:

.. code-block:: sh

   input_dir=/path/to/fmriprep
   output_dir=/path/to/output

   apptainer run --cleanenv \
     -B ${input_dir}:/input_dir \
     -B ${output_dir}:/output_dir \
     dR2star.sif \
     /input_dir /output_dir participant \
     --space MNI152NLin6Asym:res-2

**Scenario 2: One common-space reference mask for all subjects/sessions**

You have a single reference mask in a common space (for example MNI152NLin6Asym:res-2)
that you want applied to every subject and session. Provide the mask file via
``--reference-mask-input``. This is only supported for non-native spaces.

Example reference mask layout:

.. code-block:: text

   /path/to/reference_masks/
     group_reference_mask_space-MNI152NLin6Asym_res-2.nii.gz

Example command:

.. code-block:: sh

   input_dir=/path/to/fmriprep
   output_dir=/path/to/output
   reference_mask=/path/to/reference_masks/group_reference_mask_space-MNI152NLin6Asym_res-2.nii.gz

   apptainer run --cleanenv \
     -B ${input_dir}:/input_dir \
     -B ${output_dir}:/output_dir \
     -B ${reference_mask}:/reference_mask.nii.gz \
     dR2star.sif \
     /input_dir /output_dir participant \
     --space MNI152NLin6Asym:res-2 \
     --reference-mask-input /reference_mask.nii.gz

**Scenario 3A: Custom reference mask per subject/session in MNI space**

You derived a custom reference mask for each subject and session in a common space
such as MNI. Provide a derivatives-style directory and set ``--reference-mask-input``
to that directory.

Example reference mask directory layout:

.. code-block:: text

   /path/to/reference_masks/
     sub-001/
       ses-01/
         anat/
           sub-001_ses-01_space-MNI152NLin6Asym_res-2_desc-reference_mask.nii.gz

Example command:

.. code-block:: sh

   input_dir=/path/to/fmriprep
   output_dir=/path/to/output
   reference_mask_dir=/path/to/reference_masks

   apptainer run --cleanenv \
     -B ${input_dir}:/input_dir \
     -B ${output_dir}:/output_dir \
     -B ${reference_mask_dir}:/reference_masks \
     dR2star.sif \
     /input_dir /output_dir participant \
     --space MNI152NLin6Asym:res-2 \
     --reference-mask-input /reference_masks

**Scenario 3B: Custom reference mask per subject/session in native space (T1w/T2w)**

You derived a custom reference mask for each subject and session in native space.
Provide a derivatives-style directory and set ``--space`` to the native space.
Native-space masks must already align to the corresponding fMRIPrep anatomy.

Example reference mask directory layout:

.. code-block:: text

   /path/to/reference_masks/
     sub-001/
       ses-01/
         anat/
           sub-001_ses-01_space-T1w_desc-reference_mask.nii.gz

Example command:

.. code-block:: sh

   input_dir=/path/to/fmriprep
   output_dir=/path/to/output
   reference_mask_dir=/path/to/reference_masks

   apptainer run --cleanenv \
     -B ${input_dir}:/input_dir \
     -B ${output_dir}:/output_dir \
     -B ${reference_mask_dir}:/reference_masks \
     dR2star.sif \
     /input_dir /output_dir participant \
     --space T1w \
     --reference-mask-input /reference_masks

How dR2star Uses fMRIPrep Outputs
---------------------------------
dR2star follows a consistent pattern to discover and process inputs:

1. It searches each subject/session ``func/`` directory for confounds files that
   match ``*_desc-confounds_timeseries.tsv`` or ``*_desc-confounds_regressors.tsv``.
2. For each confounds file, it derives the matching preprocessed BOLD path by
   replacing the confounds suffix with ``_space-<space>_desc-preproc_bold.nii.gz``.
3. It reads framewise displacement (and optional DVARS) from the confounds file to
   build a censor mask, then applies ``--maxvols`` and ``--sample-method`` if supplied.
4. It merges the selected volumes into a single intermediate BOLD file per group.
5. If ``--reference-mask-input`` is not provided, it uses the fMRIPrep brain mask
   in the ``func/`` directory (``*_space-<space>_desc-brain_mask.nii.gz``). If a
   reference mask is provided, it uses that file or directory instead.
6. When concatenating runs, the reference mask associated with the run that has the
   most selected volumes is used, and masks are resampled to the merged BOLD grid
   when needed using nearest-neighbor interpolation.
