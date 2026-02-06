dR2star
=======

dR2star is a BIDS-app style pipeline designed to calculate estimates
of T2* based on fMRIPrep outputs. It wraps the ``dr2`` processing
binary and scans each ``func/`` directory to aggregate matching
preprocessed BOLD runs before generating outputs.

Getting started
---------------
Before running dR2star, you will need single-echo fMRI data and a completed
fMRIPrep run. dR2star operates on fMRIPrep outputs, so make sure your dataset
is BIDS-formatted and has been preprocessed with fMRIPrep. If you are new to
fMRIPrep, see the `fMRIPrep documentation <https://fmriprep.org/en/stable/>`_
for setup, inputs, and example workflows.

The pages below cover the most common next steps:

- :doc:`container` for how to pull and run the container image. This is the recommended way to use dR2star.
- :doc:`usage` for CLI options and processing configuration.
- :doc:`expected_outputs` for what results to expect and how file names change.


.. toctree::
   :maxdepth: 1
   :caption: Documentation Contents:

   Home <self>
   usage
   container
   expected_outputs
