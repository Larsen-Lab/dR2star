#!/usr/bin/env python3
"""Argument parsing for the dR2star wrapper."""

from __future__ import annotations

import argparse
import textwrap


def _parse_bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: '{value}'.")


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
        "--average",
        dest="average",
        metavar="ENTITY",
        nargs="+",
        choices=["acq", "rec", "dir", "run", "echo", "part", "ce"],
        help=(
            "Average volumes across the selected BIDS entities "
            "(acq, rec, dir, run, echo, part, ce)."
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
        "--voxscale",
        dest="voxscale",
        metavar="BOOL",
        type=_parse_bool,
        help=(
            "Override method-derived voxel scaling choice. "
            "Accepts true/false."
        ),
    )
    parser.add_argument(
        "--inverse",
        dest="inverse",
        action="store_true",
        help="Output R2* (i.e., 1/T2*) instead of T2* (tat2 -inverse).",
    )
    parser.add_argument(
        "--time-norm",
        dest="time_norm",
        choices=["none", "mean", "median"],
        default="median",
        help=(
            "Time normalization method (tat2 -mean_time/-median_time). "
            "Use 'none' for default behavior."
        ),
    )
    parser.add_argument(
        "--volume-norm",
        dest="volume_norm",
        choices=["none", "mean", "median"],
        default="median",
        help=(
            "Volume normalization method (tat2 -mean_vol/-median_vol/-no_vol). "
            "Use 'none' to disable volume normalization."
        ),
    )
    parser.add_argument(
        "--method",
        dest="method",
        choices=["neglog", "signalproportion", "zsignalproportion"],
        default="neglog",
        help=(
            "Computation method for the dR2star map. "
            "Choices: neglog, signalproportion, zsignalproportion."
        ),
    )
    parser.add_argument(
        "--use-ln",
        dest="use_ln",
        metavar="BOOL",
        type=_parse_bool,
        help=(
            "Override method-derived log transform choice. "
            "Accepts true/false."
        ),
    )
    parser.add_argument(
        "--use-zscore",
        dest="use_zscore",
        metavar="BOOL",
        type=_parse_bool,
        help=(
            "Override method-derived z-score choice. "
            "Accepts true/false."
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
