#!/usr/bin/env python3
"""BIDS-style wrapper to run tat2 with the fmriprep workflow."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from . import utilities
from .my_parser import get_parser


def build_cmd_template(
    bold_path: Path,
    censor_output_path: Path,
    mask_path: Path,
    output_path: Path,
    args,
) -> list[str]:
    """Build a minimal tat2 command for a single run."""
    cmd = [
        "tat2",
        str(bold_path),
        "-censor_rel",
        str(censor_output_path),
        "-mask",
        str(mask_path),
        "-output",
        str(output_path),
    ]
    if args.scale is not None:
        cmd.extend(["-scale", str(args.scale)])
    if args.no_voxscale:
        cmd.append("-no_voxscale")
    if args.inverse:
        cmd.append("-inverse")
    if args.time_norm == "mean":
        cmd.append("-mean_time")
    elif args.time_norm == "median":
        cmd.append("-median_time")

    if args.volume_norm == "mean":
        cmd.append("-mean_vol")
    elif args.volume_norm == "median":
        cmd.append("-median_vol")
    elif args.volume_norm == "none":
        cmd.append("-no_vol")
    if args.maxvols is not None:
        cmd.extend(["-maxvols", str(args.maxvols)])
    if args.maxvolstotal is not None:
        cmd.extend(["-maxvolstotal", str(args.maxvolstotal)])
    if args.sample_method:
        cmd.extend(["-sample_method", args.sample_method])
    if args.tmp_dir:
        cmd.extend(["-tmp", args.tmp_dir])
    if args.noclean:
        cmd.append("-noclean")
    if args.verbose:
        cmd.append("-verbose")
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = get_parser()
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    utilities.ensure_dataset_description(output_dir)

    participant_labels = utilities._normalize_labels(args.participant_label or [], "sub-")
    ses_labels = utilities._normalize_labels(args.ses_label or [], "ses-")

    if participant_labels:
        print(f"Participants: {', '.join(participant_labels)}")
    else:
        print("Participants: all")
    if ses_labels:
        print(f"Sessions: {', '.join(ses_labels)}")
    else:
        print("Sessions: all")

    env = os.environ.copy()
    if args.fd_thres is not None:
        env["FD_THRES"] = str(args.fd_thres)

    subjects = utilities._discover_subjects(input_dir, participant_labels)
    print(f"Found a total of {len(subjects)} subjects that will be considered for processing.")
    #Iterate through all possible subjects
    for subject_idx, temp_subject in enumerate(subjects, start=1):
        print(f"Subject [{subject_idx}/{len(subjects)}]: sub-{temp_subject}")
        subject_dir = input_dir / f"sub-{temp_subject}"
        sessions = utilities._discover_sessions(subject_dir, ses_labels)
        #Iterate through all possible sessions
        for temp_session in sessions:
            session_label = f"ses-{temp_session}" if temp_session else "ses-<none>"
            print(f"Session: {session_label}")
            if temp_session:
                func_directory = subject_dir / f"ses-{temp_session}" / "func"
                confound_patterns = [
                    f"sub-{temp_subject}_ses-{temp_session}_*desc-confounds_timeseries.tsv",
                    f"sub-{temp_subject}_ses-{temp_session}_*desc-confounds_regressors.tsv",
                ]
                output_anat_dir = (
                    output_dir
                    / f"sub-{temp_subject}"
                    / f"ses-{temp_session}"
                    / "anat"
                )
            else:
                func_directory = subject_dir / "func"
                confound_patterns = [
                    f"sub-{temp_subject}_*desc-confounds_timeseries.tsv",
                    f"sub-{temp_subject}_*desc-confounds_regressors.tsv",
                ]
                output_anat_dir = output_dir / f"sub-{temp_subject}" / "anat"

            #Find all confound files for this subject/session. There
            #will be one confound file per fMRI acquisition.
            confound_files: list[Path] = []
            for pattern in confound_patterns:
                confound_files.extend(func_directory.glob(pattern))
            confound_files = sorted({path for path in confound_files})

            #If the user specified specific task labels, be sure to only keep those
            #confound files that match the requested tasks.
            if args.task_id:
                task_ids = [task.removeprefix("task-") for task in args.task_id]
                confound_files = [
                    path
                    for path in confound_files
                    if any(f"_task-{task}_" in path.name for task in task_ids)
                ]

            output_anat_dir.mkdir(parents=True, exist_ok=True)

            #For every confound file, try to run the dR2star pipeline.
            for confound_file in confound_files:
                print(f"Run: {confound_file.name}")

                #Populate the corresponding output censor file name.
                confound_name = confound_file.name
                censor_name = utilities._replace_confounds_suffix(
                    confound_name,
                    "_desc-dR2star_censor.1D",
                )
                censor_output_path = output_anat_dir / censor_name
                
                #Create the censor file from the confounds and the given thresholds.
                utilities.confounds_to_censor_file(
                    confound_file,
                    str(censor_output_path),
                    fd_thres=args.fd_thres,
                    dvars_thresh=args.dvars_thresh,
                )

                space_token = args.space.replace(":", "_")
                bold_name = utilities._replace_confounds_suffix(
                    confound_name,
                    f"_space-{space_token}_desc-preproc_bold.nii.gz",
                )
                bold_path = func_directory / bold_name
                if not bold_path.exists():
                    raise FileNotFoundError(
                        f"Missing preproc bold file for space '{args.space}': {bold_path}"
                    )

                if args.mask_input is None:
                    mask_name = utilities._replace_confounds_suffix(
                        confound_name,
                        f"_space-{space_token}_desc-brain_mask.nii.gz",
                    )
                    mask_path = func_directory / mask_name
                else:
                    mask_input = Path(args.mask_input)
                    if mask_input.is_file():
                        mask_path = mask_input
                    elif mask_input.is_dir():
                        raise NotImplementedError(
                            "Mask derivatives directory support is not implemented yet."
                        )
                    else:
                        raise FileNotFoundError(
                            f"--mask-input does not exist: {args.mask_input}"
                        )

                if not bold_name.endswith("_desc-preproc_bold.nii.gz"):
                    raise NameError(f"Unexpected bold file name format: {bold_name}")
                output_name = bold_name.replace(
                    "_desc-preproc_bold.nii.gz",
                    "_desc-dR2star_dR2starmap.nii.gz",
                )
                output_path = output_anat_dir / output_name

                cmd_template = build_cmd_template(
                    bold_path,
                    censor_output_path,
                    mask_path,
                    output_path,
                    args,
                )
                try:
                    result = subprocess.run(cmd_template, check=False, env=env)
                except FileNotFoundError:
                    parser.error("'tat2' not found on PATH. Ensure it is installed or in PATH.")

                if result.returncode != 0:
                    return result.returncode

                log_json = Path(str(output_path).replace(".nii.gz", ".log.json"))
                if log_json.exists():
                    sidecar_json = Path(str(output_path).replace(".nii.gz", ".json"))
                    log_json.replace(sidecar_json)
                    utilities.postprocess_tat2_json(
                        sidecar_json,
                        input_dir,
                        output_dir,
                        confound_file,
                        args.fd_thres,
                        args.dvars_thresh,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
