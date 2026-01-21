#!/usr/bin/env python3
"""Argument parsing for the dR2star wrapper."""

from __future__ import annotations

import argparse
import textwrap


def get_parser() -> argparse.ArgumentParser:
    description = """
    dR2star wrapper for tat2 fmriprep runs.

    This interface mirrors a BIDS App-style CLI with three positional
    arguments: input, output, and analysis level. Only the participant
    analysis level is supported.
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
            "This is passed to tat2 using -fmriprep."
        ),
    )
    parser.add_argument(
        "output_dir",
        metavar="OUTPUT_DIR",
        help=(
            "Root output directory. Outputs are written under "
            "OUTPUT_DIR/sub-<label>/ses-<label>/anat/."
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
            "separated by spaces (e.g., 'rest' 'nback')."
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
        "--mask-input",
        dest="mask_input",
        metavar="PATH",
        help=(
            "Mask input: either a derivatives-like directory containing per-subject/session masks "
            "or a single mask file in standard space to apply to all subjects."
        ),
    )
    parser.add_argument(
        "--scale",
        dest="scale",
        metavar="SCALE",
        type=float,
        help="Scale factor passed to tat2 (-scale).",
    )
    parser.add_argument(
        "--no-voxscale",
        dest="no_voxscale",
        action="store_true",
        help="Disable voxel scaling (tat2 -no_voxscale).",
    )
    parser.add_argument(
        "--inverse",
        dest="inverse",
        action="store_true",
        help="Use 1/T2* input (tat2 -inverse).",
    )
    parser.add_argument(
        "--time-norm",
        dest="time_norm",
        choices=["none", "mean", "median"],
        default="none",
        help=(
            "Time normalization method (tat2 -mean_time/-median_time). "
            "Use 'none' for default behavior."
        ),
    )
    parser.add_argument(
        "--volume-norm",
        dest="volume_norm",
        choices=["none", "mean", "median"],
        default="none",
        help=(
            "Volume normalization method (tat2 -mean_vol/-median_vol/-no_vol). "
            "Use 'none' to disable volume normalization."
        ),
    )
    parser.add_argument(
        "--fd-thres",
        dest="fd_thres",
        metavar="THRESH",
        type=float,
        default=0.3,
        help="Framewise displacement threshold for fmriprep confounds (FD_THRES env).",
    )
    parser.add_argument(
        "--dvars-thresh",
        dest="dvars_thresh",
        metavar="THRESH",
        type=float,
        help="DVARS threshold for confounds filtering (currently unused).",
    )
    parser.add_argument(
        "-w",
        dest="tmp_dir",
        metavar="DIR",
        help="Working directory for intermediate files (tat2 -tmp).",
    )
    parser.add_argument(
        "--noclean",
        dest="noclean",
        action="store_true",
        help="Keep temporary files (tat2 -noclean).",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Enable verbose logging (tat2 -verbose).",
    )

    return parser
