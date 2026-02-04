Usage
=====
.. argparse::
   :ref: dR2star.my_parser.get_parser
   :prog: dR2star
   :nodefaultconst:

Masking Details
---------------
This workflow supports three masking modes, depending on the ``--mask-input`` flag:

1) **No** ``--mask-input`` **provided**: A per-run fMRIPrep brain mask is used from
   the input directory (``*_space-<space>_desc-brain_mask.nii.gz``). If multiple runs
   are to be concatenated, the mask from the run with the most thresholded volumes will
   be used for all runs.

2) **Single mask file**: Provide a single mask path to apply to all runs. This
   is only supported for non-native spaces (e.g., MNI). For native spaces
   (T1w/T2w), a single custom mask is rejected.

3) **Mask directory**: Provide a derivatives-like directory that mirrors the
   input layout (``sub-<id>/[ses-<id>/]anat``). The workflow searches for exactly
   one mask per subject/session using the pattern:

   ``sub-<id>[_ses-<id>]*_space-<space>[_res-<res>]_desc-brain_mask.nii.gz``

   If zero or multiple masks match, the run stops with a descriptive error. If the mask
   lives on a different grid than the merged BOLD data, it is resampled using nearest-neighbor
   interpolation. The resampled mask is saved alongside the outputs in the ``anat`` folder and
   is only written when resampling is needed. The output JSON includes a ``mask_resampled`` field to
   indicate whether resampling occurred.

   Similar to case (1), if multiple runs are to be concatenated, the mask will be resampled
   to the space of the fMRI run with the most volumes.
   
   **Warning:** For native spaces (T1w/T2w), the pipeline assumes masks are already aligned
   to the corresponding preprocessed T1w/T2w anatomy in the fMRIPrep ``anat`` directory.

   Example directory layouts:

   - **Default MNI (with resolution tag)**::

       mask_dir/
         sub-001/
           ses-01/
             anat/
               sub-001_ses-01_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz

   - **T1w native space**::

       mask_dir/
         sub-001/
           ses-01/
             anat/
               sub-001_ses-01_space-T1w_desc-brain_mask.nii.gz
