#!/usr/bin/env python3
"""Argument parsing for the dR2star wrapper."""

from __future__ import annotations

import argparse
import textwrap


_METHOD_CHOICES = ("dR2star", "nT2star", "zscore")
_METHOD_ALIASES = {
    "dr2star": "dR2star",
    "nt2star": "nT2star",
    "zscore": "zscore",
}


def _parse_method(value: str) -> str:
    key = value.strip().lower()
    if key in _METHOD_ALIASES:
        return _METHOD_ALIASES[key]
    choices = ", ".join(_METHOD_CHOICES)
    raise argparse.ArgumentTypeError(
        f"Invalid method '{value}'. Choices (case-insensitive): {choices}."
    )


def get_parser() -> argparse.ArgumentParser:
    description = """
    dR2star is a BIDS-App designed to generate T2* estimates using
    single-echo fMRI outputs from fMRIPrep. It generates dR2* maps
    from those preprocessed outputs.

    This interface mirrors a BIDS App-style CLI with three positional
    arguments: input, output, and analysis level.
    """

    epilog = """
    Examples
    --------
    Process all participants and sessions:
      dR2star /data/derivatives/fmriprep /data/derivatives/dR2star participant

    Process a single participant:
      dR2star /data/derivatives/fmriprep /data/derivatives/dR2star participant \
        --participant-label 01

    Process a single participant/session:
      dR2star /data/derivatives/fmriprep /data/derivatives/dR2star participant \
        --participant-label 01 --ses-label 02
    """

    parser = argparse.ArgumentParser(
        prog="dR2star",
        description=textwrap.dedent(description).strip(),
        epilog=textwrap.dedent(epilog).strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input_dir",
        metavar="INPUT_DIR",
        help=(
            "Path to fmriprep derivatives (input directory). "
        ),
    )
    parser.add_argument(
        "output_dir",
        metavar="OUTPUT_DIR",
        help=(
            "Root output directory. Outputs are written under "
            "OUTPUT_DIR/sub-<label>[/ses-<label>]/anat/."
        ),
    )
    parser.add_argument(
        "analysis_level",
        metavar="ANALYSIS_LEVEL",
        choices=["participant"],
        help="Processing level to run. Only 'participant' is supported.",
    )

    parser.add_argument(
        "--participant-label",
        dest="participant_label",
        metavar="LABEL",
        nargs="+",
        help=(
            "Optional participant label(s) (with or without 'sub-'). "
            "Provide one or more labels separated by spaces. "
            "Example: '01' or 'sub-01'."
        ),
    )
    parser.add_argument(
        "--ses-label",
        dest="ses_label",
        metavar="LABEL",
        nargs="+",
        help=(
            "Optional session label(s) (with or without 'ses-'). "
            "Provide one or more labels separated by spaces. "
            "Example: '01' or 'ses-01'."
        ),
    )
    parser.add_argument(
        "-t",
        "--task-id",
        dest="task_id",
        metavar="TASK",
        nargs="+",
        help=(
            "Optional task ID(s) to process. Provide one or more task IDs "
            "separated by spaces (e.g., 'rest' 'nback'). "
            "By default all tasks are processed."
        ),
    )
    parser.add_argument(
        "--dr2star-method",
        dest="method",
        type=_parse_method,
        choices=_METHOD_CHOICES,
        default="dR2star",
        help=(
            "Computation method for the dR2star map. "
            "Choices (case-insensitive): dR2star, nT2star, zscore."
        ),
    )
    parser.add_argument(
        "--space",
        dest="space",
        metavar="SPACE",
        default="MNI152NLin6Asym:res-2",
        help=(
            "Volumetric space specifier for fMRIPrep outputs. "
            "Provide a single value (e.g., 'MNI152NLin6Asym:res-2', 'T1w')."
        ),
    )
    parser.add_argument(
        "--reference-mask-input",
        dest="reference_mask_input",
        metavar="PATH",
        help=(
            "Reference mask input: either a derivatives-like directory containing per-subject/session "
            "reference masks or a single reference mask file in standard space to apply to all subjects. "
            "This mask defines the reference region used in normalization during dR2* computation."
        ),
    )
    parser.add_argument(
        "--concat",
        dest="concat",
        metavar="ENTITY",
        nargs="+",
        choices=["acq", "rec", "dir", "run", "echo", "part", "ce"],
        help=(
            "Concatenate volumes across the selected BIDS entities "
            "prior to dR2* computation. Accepts one or more of"
            "acq, rec, dir, run, echo, part, ce."
        ),
    )
    parser.add_argument(
        "--keep-merged",
        dest="keep_merged",
        action="store_true",
        help="Keep merged intermediate volumes used for dR2* computation.",
    )
    parser.add_argument(
        "--scale",
        dest="scale",
        metavar="SCALE",
        type=float,
        help="Scale factor applied during normalization.",
    )
    parser.add_argument(
        "--time-average-method",
        dest="time_norm",
        choices=["mean", "median"],
        default="median",
        help="Time averaging method.",
    )
    parser.add_argument(
        "--reference-average-method",
        dest="volume_norm",
        choices=["mean", "median"],
        default="median",
        help="Reference region averaging method.",
    )
    parser.add_argument(
        "--maxvols",
        dest="maxvols",
        metavar="NVOL",
        type=int,
        help="Limit total selected volumes across all runs in a group of concatenated files.",
    )
    parser.add_argument(
        "--sample-method",
        dest="sample_method",
        choices=["first", "last", "random"],
        help="Sub-sampling method for confounds-based selection across runs.",
    )
    parser.add_argument(
        "--fd-thres",
        dest="fd_thres",
        metavar="THRESH",
        type=float,
        default=0.3,
        help="Framewise displacement threshold for confounds filtering.",
    )
    parser.add_argument(
        "--dvars-thresh",
        dest="dvars_thresh",
        metavar="THRESH",
        type=float,
        help="DVARS threshold for confounds filtering (omit to disable).",
    )
    parser.add_argument(
        "-w",
        dest="tmp_dir",
        metavar="DIR",
        help="Working directory for intermediate files.",
    )
    parser.add_argument(
        "--noclean",
        dest="noclean",
        action="store_true",
        help="Keep temporary files.",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    return parser
