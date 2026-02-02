#!/usr/bin/env python3
"""BIDS-style wrapper to run tat2 with the fmriprep workflow."""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path

from . import utilities
from .my_parser import get_parser


def build_cmd_template(
    bold_path: Path,
    censor_output_path: Path | None,
    mask_path: Path,
    output_path: Path,
    args,
) -> list[str]:
    """Build a minimal tat2 command for a single run."""
    cmd = [
        "tat2",
        str(bold_path),
    ]
    if censor_output_path:
        cmd.extend(["-censor_rel", str(censor_output_path)])
    cmd.extend(["-mask", str(mask_path), "-output", str(output_path)])
    if args.scale is not None:
        cmd.extend(["-scale", str(args.scale)])
    if not args.voxscale:
        cmd.append("-no_voxscale")
    if args.inverse:
        cmd.append("-inverse")
    if args.time_norm == "mean":
        cmd.append("-mean_time")
    elif args.time_norm == "median":
        cmd.append("-median_time")

    if args.use_ln:
        cmd.append("-calc_ln")
    elif args.use_zscore:
        cmd.append("-calc_zscore")

    if args.volume_norm == "mean":
        cmd.append("-mean_vol")
    elif args.volume_norm == "median":
        cmd.append("-median_vol")
    elif args.volume_norm == "none":
        cmd.append("-no_vol")
    if args.tmp_dir:
        cmd.extend(["-tmp", args.tmp_dir])
    else:
        cmd.extend(["-tmp", str(output_path.parent)])
    if args.noclean:
        cmd.append("-noclean")
    if args.verbose:
        cmd.append("-verbose")
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = get_parser()
    args = parser.parse_args(argv)
    if args.method == "neglog":
        default_use_ln = True
        default_use_zscore = False
        default_voxscale = False
    elif args.method == "signalproportion":
        default_use_ln = False
        default_use_zscore = False
        default_voxscale = False
    elif args.method == "zsignalproportion":
        default_use_ln = False
        default_use_zscore = True
        default_voxscale = False
    else:
        parser.error(f"Unsupported method '{args.method}'")
    use_ln = default_use_ln
    use_zscore = default_use_zscore
    voxscale = default_voxscale
    if args.use_ln is not None:
        if args.use_ln != default_use_ln:
            print(
                "Warning: method chosen was "
                f"{args.method}, which has default value for use_ln "
                f"as {default_use_ln}, but this option is being changed "
                f"to {args.use_ln} based on user-defined processing flags."
            )
        use_ln = args.use_ln
        if args.use_ln and args.use_zscore is None:
            use_zscore = False
    if args.use_zscore is not None:
        if args.use_zscore != default_use_zscore:
            print(
                "Warning: method chosen was "
                f"{args.method}, which has default value for use_zscore "
                f"as {default_use_zscore}, but this option is being changed "
                f"to {args.use_zscore} based on user-defined processing flags."
            )
        use_zscore = args.use_zscore
        if args.use_zscore and args.use_ln is None:
            use_ln = False
    if args.voxscale is not None:
        if args.voxscale != default_voxscale:
            print(
                "Warning: method chosen was "
                f"{args.method}, which has default value for voxscale "
                f"as {default_voxscale}, but this option is being changed "
                f"to {args.voxscale} based on user-defined processing flags."
            )
        voxscale = args.voxscale
    if use_ln and use_zscore:
        parser.error("--use-ln and --use-zscore cannot both be true.")
    if voxscale and (use_ln or use_zscore):
        parser.error(
            "--voxscale cannot be combined with --use-ln/--use-zscore or a method "
            "that implies them; tat2 requires -no_voxscale for those transforms."
        )
    args.use_ln = use_ln
    args.use_zscore = use_zscore
    args.voxscale = voxscale

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    utilities.ensure_dataset_description(output_dir)
    env = os.environ.copy()
    if args.fd_thres is not None:
        env["FD_THRES"] = str(args.fd_thres)

    def to_bids_uri(path: Path) -> str:
        return (
            str(path)
            .replace(str(input_dir), "bids:preprocessed:")
            .replace(str(output_dir), "bids::")
        )

    def write_json_with_inline_masks(path: Path, payload: dict) -> None:
        mask_tokens: dict[str, list[int]] = {}

        def _mark_masks(obj):
            if isinstance(obj, dict):
                marked: dict = {}
                for key, value in obj.items():
                    if key == "mask" and isinstance(value, list):
                        token = f"__MASK_INLINE_{uuid.uuid4().hex}__"
                        mask_tokens[token] = [int(v) for v in value]
                        marked[key] = token
                    else:
                        marked[key] = _mark_masks(value)
                return marked
            if isinstance(obj, list):
                return [_mark_masks(item) for item in obj]
            return obj

        marked_payload = _mark_masks(payload)
        json_text = json.dumps(marked_payload, indent=2, sort_keys=True)
        for token, mask in mask_tokens.items():
            mask_text = "[" + ", ".join(str(v) for v in mask) + "]"
            json_text = json_text.replace(f"\"{token}\"", mask_text)
        path.write_text(json_text + "\n")

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


    space_token = args.space.replace(":", "_")
    if (space_token == "T1w") or (space_token == "T2w"):
        is_native = True
    else:
        is_native = False

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
            print(
                f"Found {len(confound_files)} confound file(s) for session {session_label}."
            )

            #For every confound file, try to run the dR2star pipeline.
            confound_names: list[Path] = []
            bold_paths: list[Path] = []
            mask_paths: list[Path] = []
            for confound_file in confound_files:
                confound_name = confound_file.name
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
                        if is_native:
                            raise ValueError(
                                "Single custom mask files are only supported for "
                                "non-native spaces. Use a derivatives mask or "
                                "choose a non-native space."
                            )
                        mask_path = mask_input
                    elif mask_input.is_dir():
                        raise NotImplementedError(
                            "Mask derivatives directory support is not implemented yet."
                        )
                    else:
                        raise FileNotFoundError(
                            f"--mask-input does not exist: {args.mask_input}"
                        )
                confound_names.append(confound_file)
                bold_paths.append(bold_path)
                mask_paths.append(mask_path)

            group_ids, num_groups, reduced_names = utilities.group_confounds_by_entities(
                args.concat or [],
                confound_files,
            )
            print(f"Grouping into {num_groups} run group(s) using --concat.")

            for group_idx in range(num_groups):
                print(f"Processing group {group_idx + 1}/{num_groups}.")
                group_confound_files = [
                    path
                    for path, gid in zip(confound_names, group_ids)
                    if gid == group_idx
                ]
                if not group_confound_files:
                    continue
                group_bold_paths = [
                    path
                    for path, gid in zip(bold_paths, group_ids)
                    if gid == group_idx
                ]
                group_mask_paths = [
                    path
                    for path, gid in zip(mask_paths, group_ids)
                    if gid == group_idx
                ]
                group_reduced_names = [
                    name
                    for name, gid in zip(reduced_names, group_ids)
                    if gid == group_idx
                ]
                if not group_bold_paths or not group_mask_paths:
                    continue
                selections = utilities.build_volume_selection_from_confounds(
                    group_confound_files,
                    group_bold_paths,
                    fd_thres=args.fd_thres,
                    dvars_thresh=args.dvars_thresh,
                    sample_method=args.sample_method,
                    maxvols=args.maxvols,
                )
                print("Merge inputs and selected volume counts:")
                for path in group_bold_paths:
                    mask = selections.get(path, [])
                    print(f"  - {path.name}: {sum(mask)} volume(s)")
                best_bold_path = max(selections, key=lambda path: len(selections[path]))
                try:
                    best_index = [
                        Path(path) for path in group_bold_paths
                    ].index(Path(best_bold_path))
                except ValueError:
                    raise ValueError(
                        "Unable to match selected volume source to a mask path."
                    )
                mask_path = group_mask_paths[best_index]
                if len({str(path) for path in group_mask_paths}) > 1:
                    print(
                        "Warning: multiple mask files found for a merged group; "
                        f"using {mask_path.name} from the run with the most volumes."
                    )
                total_kept = sum(sum(mask) for mask in selections.values())
                print(
                    f"Selected {total_kept} total volume(s) across "
                    f"{len(group_bold_paths)} run(s)."
                )
                reduced_name = group_reduced_names[0]
                merged_bold_name = utilities._replace_confounds_suffix(
                    reduced_name,
                    f"_space-{space_token}_desc-MergedIntermediate_bold.nii.gz",
                )
                merged_output_path = output_anat_dir / merged_bold_name
                print(f"Writing merged intermediate: {merged_output_path.name}")
                utilities.merge_selected_volumes(
                    selections,
                    merged_output_path,
                    needs_resampling=is_native,
                )
                print("Merged intermediate complete.")

                selection_sources = [to_bids_uri(path) for path in group_bold_paths]
                volume_selection: dict[str, dict[str, list[int]]] = {}
                for path, mask in selections.items():
                    volume_selection[to_bids_uri(Path(path))] = {
                        "mask": mask,
                    }
                selection_metadata = {
                    "source_data": selection_sources,
                    "volume_selection": volume_selection,
                    "selection_params": {
                        "fd_thres": args.fd_thres,
                        "dvars_thresh": args.dvars_thresh,
                        "sample_method": args.sample_method or "first",
                        "maxvols": args.maxvols,
                    },
                }
                merged_json_path = Path(
                    str(merged_output_path).replace(".nii.gz", ".json")
                )
                write_json_with_inline_masks(merged_json_path, selection_metadata)

                if not merged_bold_name.endswith("_desc-MergedIntermediate_bold.nii.gz"):
                    raise NameError(
                        f"Unexpected merged bold file name format: {merged_bold_name}"
                    )
                output_name = merged_bold_name.replace(
                    "_desc-MergedIntermediate_bold.nii.gz",
                    "_desc-dR2star_dR2starmap.nii.gz",
                )
                output_path = output_anat_dir / output_name
                print(f"Running tat2 for: {output_path.name}")

                cmd_template = build_cmd_template(
                    merged_output_path,
                    None,
                    mask_path,
                    output_path,
                    args,
                )
                try:
                    result = subprocess.run(
                        cmd_template,
                        check=False,
                        env=env,
                        cwd=output_anat_dir,
                    )
                except FileNotFoundError:
                    parser.error("'tat2' not found on PATH. Ensure it is installed or in PATH.")

                if result.returncode != 0:
                    return result.returncode
                print("tat2 complete.")

                log_json = Path(str(output_path).replace(".nii.gz", ".log.json"))
                if log_json.exists():
                    sidecar_json = Path(str(output_path).replace(".nii.gz", ".json"))
                    log_json.replace(sidecar_json)
                    utilities.postprocess_tat2_json(
                        sidecar_json,
                        input_dir,
                        output_dir,
                        group_confound_files,
                        args.fd_thres,
                        args.dvars_thresh,
                    )
                    data = json.loads(sidecar_json.read_text())
                    data["volume_selection"] = selection_metadata["volume_selection"]
                    data["selection_params"] = selection_metadata["selection_params"]
                    data["source_data"] = selection_metadata["source_data"]
                    write_json_with_inline_masks(sidecar_json, data)
                if not args.keep_merged:
                    print(f"Removing merged intermediate: {merged_output_path.name}")
                    merged_output_path.unlink(missing_ok=True)
                    merged_json_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
