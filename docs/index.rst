dR2star
=======

dR2star is a BIDS-app style pipeline designed to calculate estimates
of T2* based on fMRIPREP outputs.

Quick start
-----------

.. code-block:: sh

   dR2star /path/to/fmriprep /path/to/output participant

See the :doc:`usage` page for the full CLI help and options.

The ``--concat`` option can be used to request concatenation across selected BIDS
entities.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   usage
   container
   expected_outputs
