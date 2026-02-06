Expected Outputs
================
This page describes what the output directory should look like and how it
changes based on key options (concatenation, space, and masking-related outputs).

Run examples
------------
Below are two small examples that show what outputs look like with and without
concatenation. Filenames include the full MNI space label and ``res-2``.

Two runs without concatenation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you do not use ``--concat``, each run produces its own dR2* map and JSON
sidecar. The run entity remains in the filename.

.. code-block:: text

   /path/to/output/
     sub-001/
       ses-01/
         anat/
           sub-001_ses-01_task-rest_run-01_space-MNI152NLin6Asym_res-2_desc-dR2star_dR2starmap.nii.gz
           sub-001_ses-01_task-rest_run-01_space-MNI152NLin6Asym_res-2_desc-dR2star_dR2starmap.json
           sub-001_ses-01_task-rest_run-02_space-MNI152NLin6Asym_res-2_desc-dR2star_dR2starmap.nii.gz
           sub-001_ses-01_task-rest_run-02_space-MNI152NLin6Asym_res-2_desc-dR2star_dR2starmap.json

Concatenated across runs
^^^^^^^^^^^^^^^^^^^^^^^^
If you concatenate across runs (``--concat run``), the run entity is dropped and
a single output is created for the concatenated group.

.. code-block:: text

   /path/to/output/
     sub-001/
       ses-01/
         anat/
           sub-001_ses-01_task-rest_space-MNI152NLin6Asym_res-2_desc-dR2star_dR2starmap.nii.gz
           sub-001_ses-01_task-rest_space-MNI152NLin6Asym_res-2_desc-dR2star_dR2starmap.json

Whether or not concatenation is active, a merged intermediate BOLD file is created. This contains
the fMRI volumes that will be used to create the dR2star map file, and is the result of censoring
(high FD/DVARS) and concatenating. If ``--keep-merged`` is set, you will also see:

.. code-block:: text

   sub-001_ses-01_task-rest_space-MNI152NLin6Asym_res-2_desc-MergedIntermediate_bold.nii.gz

If ``--keep-merged`` is not set, that merged intermediate is deleted after
processing.

What else can be concatenated?
------------------------------
The ``--concat`` flag can group volumes across other BIDS entities, not just
``run``. Supported entities are: ``acq``, ``rec``, ``dir``, ``run``, ``echo``,
``part``, and ``ce``. You may pass multiple values. For example:

- ``--concat run task`` merges across runs and tasks.

In general, the entities you concatenate over are removed from the output
filename to reflect the merged group.

**Warning:** Only concatenate entities that have compatible acquisition
parameters. For example, you may not want to merge runs with different voxel sizes, echo
times, etc.

Mask-related outputs
--------------------
If a mask needs resampling to match the merged BOLD grid, the resampled mask is
saved with a ``_desc-reference_mask`` suffix and the JSON records the original and
resampled mask paths. If resampling is not needed, no additional mask file is created.

Example resampled mask:

.. code-block:: text

   sub-001_ses-01_space-MNI152NLin6Asym_res-2_desc-reference_mask.nii.gz

Space-dependent outputs
-----------------------
The ``space-`` and ``res-`` entities reflect the requested volumetric space. For
example:

- MNI space:

  .. code-block:: text

     sub-001_ses-01_space-MNI152NLin6Asym_res-2_desc-dR2star_dR2starmap.nii.gz

- Native T1w space:

  .. code-block:: text

     sub-001_ses-01_space-T1w_desc-dR2star_dR2starmap.nii.gz

JSON contents (expanded)
------------------------
Each output ``.json`` sits next to its ``.nii.gz`` and provides provenance and
selection metadata. Paths are stored as BIDS URIs (for example ``bids::`` for
output paths and ``bids:preprocessed:`` for input paths).

Common fields include:

- ``dr2star_generated``: the raw settings and command provenance from the
  dR2star processing step (including normalization choices and runtime commands).
- ``confounds_file``: the confounds TSV(s) used to generate volume selections.
- ``fd_thres`` and ``dvars_thresh``: motion and signal quality thresholds used for selection.
- ``source_data``: list of BIDS URIs for the input BOLD files used in this
  output.
- ``volume_selection``: a mapping from each input BOLD file to its
  ``temporal_mask`` (0/1 list indicating which volumes were kept).
- ``selection_params``: the selection controls used (``sample_method`` and
  ``maxvols``).
- ``mask_file``: the final mask path used for normalization (BIDS URI).
- ``mask_resampled``: whether the mask was resampled to match the merged BOLD.
- ``mask_resample_map`` (when applicable): original and resampled mask paths.

These fields allow you to trace exactly which volumes, inputs, and masks were
used to compute each dR2* output.
