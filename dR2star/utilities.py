#!/usr/bin/env python3
import json
import os
import re
from pathlib import Path

import pandas as pd
import numpy as np


def _normalize_labels(labels: list[str], prefix: str) -> list[str]:
    """Strip a BIDS prefix from each label (e.g., sub-, ses-)."""
    return [label.removeprefix(prefix) for label in labels]


def _discover_subjects(input_dir: Path, requested: list[str]) -> list[str]:
    """Return subject labels from the input tree unless explicitly provided."""
    input_path = Path(input_dir)
    if requested:
        return requested
    subjects = sorted(
        path.name.removeprefix("sub-")
        for path in input_path.glob("sub-*")
        if path.is_dir()
    )
    if not subjects:
        raise FileNotFoundError(f"No subject directories found under {input_dir}")
    return subjects


def _discover_sessions(subject_dir: Path, requested: list[str]) -> list[str | None]:
    """Return session labels from a subject tree, or [None] when no sessions."""
    subject_path = Path(subject_dir)
    sessions = sorted(
        path.name.removeprefix("ses-")
        for path in subject_path.glob("ses-*")
        if path.is_dir()
    )
    if sessions:
        if requested:
            sessions = [session for session in sessions if session in requested]
            if not sessions:
                raise FileNotFoundError(
                    f"No requested sessions found under {subject_dir}"
                )
        return sessions
    if requested:
        raise FileNotFoundError(
            f"Session labels provided but no ses-* directories found under {subject_dir}"
        )
    return [None]


def _replace_confounds_suffix(filename: str, suffix: str) -> str:
    """Swap a confounds TSV suffix for a target suffix."""
    if filename.endswith("_desc-confounds_timeseries.tsv"):
        return filename.replace("_desc-confounds_timeseries.tsv", suffix)
    if filename.endswith("_desc-confounds_regressors.tsv"):
        return filename.replace("_desc-confounds_regressors.tsv", suffix)
    raise NameError(f"Unexpected confound file name format: {filename}")


def find_mask_in_directory(
    mask_root: Path,
    subject: str,
    session: str | None,
    space_token: str,
) -> Path:
    """Find a single mask file in a derivatives-like mask directory."""
    subject_dir = mask_root / f"sub-{subject}"
    if session:
        anat_dir = subject_dir / f"ses-{session}" / "anat"
        session_tag = f"_ses-{session}"
    else:
        anat_dir = subject_dir / "anat"
        session_tag = ""

    if not anat_dir.exists():
        raise FileNotFoundError(
            "Mask directory missing expected anat folder: "
            f"{anat_dir}. Ensure the mask directory mirrors the input layout."
        )

    base = f"sub-{subject}{session_tag}*_space-{space_token}"
    patterns = [
        f"{base}_desc-*_mask.nii.gz",
        f"{base}_mask.nii.gz",
    ]
    matches = sorted({path for pattern in patterns for path in anat_dir.glob(pattern)})
    if len(matches) != 1:
        pattern_msg = " or ".join(f"'{pattern}'" for pattern in patterns)
        raise ValueError(
            "Expected exactly 1 mask file matching "
            f"{pattern_msg} in {anat_dir}, found {len(matches)}. "
            "Ensure there is exactly one mask per subject/session/space or "
            "rename masks to match the expected pattern."
        )
    return matches[0]


def resample_mask_to_reference(
    mask_path: Path,
    reference_path: Path,
    output_dir: Path,
    output_base: Path | None = None,
) -> tuple[Path, bool]:
    """Resample a mask to a reference image grid if needed."""
    import nibabel as nib
    from nibabel import processing

    mask_img = nib.load(str(mask_path), mmap=True)
    ref_img = nib.load(str(reference_path), mmap=True)

    if mask_img.ndim > 3:
        if mask_img.shape[3] != 1:
            raise ValueError(
                f"Mask {mask_path} must be 3D or single-volume 4D, "
                f"got shape {mask_img.shape}."
            )
        mask_img = mask_img.slicer[..., 0]
    if ref_img.ndim > 3:
        ref_img = ref_img.slicer[..., 0]

    same_shape = mask_img.shape[:3] == ref_img.shape[:3]
    same_affine = np.allclose(mask_img.affine, ref_img.affine)
    if same_shape and same_affine:
        return mask_path, False

    resampled = processing.resample_from_to(mask_img, ref_img, order=0)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_base is not None:
        base_name = output_base.name
        if base_name.endswith(".nii.gz"):
            base = base_name[:-7]
        elif base_name.endswith(".nii"):
            base = base_name[:-4]
        else:
            base = output_base.stem
        if "_desc-dR2star_dR2starmap" in base:
            base = base.replace("_desc-dR2star_dR2starmap", "_desc-reference_mask")
        else:
            base = f"{base}_desc-reference_mask"
        out_name = f"{base}.nii.gz"
    elif mask_path.name.endswith(".nii.gz"):
        stem = mask_path.name[:-7]
        out_name = f"{stem}_resampled.nii.gz"
    else:
        out_name = f"{mask_path.stem}_resampled{mask_path.suffix}"
    out_path = output_dir / out_name
    nib.save(resampled, str(out_path))
    return out_path, True


def mask_path_to_uri(
    mask_path: Path,
    input_dir: Path,
    output_dir: Path,
    mask_input: str | None,
) -> str:
    """Return a BIDS-style URI for a mask path."""
    mask_path = Path(mask_path)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if mask_path.is_relative_to(output_dir):
        rel = mask_path.relative_to(output_dir)
        return f"bids::{rel.as_posix()}"
    if mask_path.is_relative_to(input_dir):
        rel = mask_path.relative_to(input_dir)
        return f"bids:preprocessed:{rel.as_posix()}"

    if mask_input is None:
        return str(mask_path)

    mask_input_path = Path(mask_input)
    if mask_input_path.is_dir():
        try:
            rel = mask_path.relative_to(mask_input_path)
            return f"bids:mask_dir:{rel.as_posix()}"
        except ValueError:
            return str(mask_path)
    if mask_input_path.is_file():
        return f"bids:mask_file:{mask_path.as_posix().lstrip('/')}"
    return str(mask_path)


def ensure_dataset_description(output_dir: Path) -> Path:
    """Create a minimal BIDS derivatives dataset_description.json if missing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    desc_path = output_dir / "dataset_description.json"
    version = os.environ.get("DR2STAR_VERSION", "unknown")
    if desc_path.exists():
        try:
            existing = json.loads(desc_path.read_text())
        except json.JSONDecodeError:
            existing = {}
    else:
        existing = {}
    description = dict(existing)
    name = description.get("Name")
    if name in ("dr2star derivatives", "dr2star"):
        description["Name"] = "dR2star derivatives"
    else:
        description.setdefault("Name", "dR2star derivatives")
    description.setdefault("BIDSVersion", "1.8.0")
    description.setdefault("DatasetType", "derivative")
    generated_by = existing.get("GeneratedBy")
    if not isinstance(generated_by, list):
        generated_by = []
    updated = False
    for entry in generated_by:
        if isinstance(entry, dict) and entry.get("Name") in ("dr2star", "dR2star"):
            entry["Name"] = "dR2star"
            entry["Version"] = version
            entry.setdefault("Description", "dR2star processing using dr2")
            updated = True
            break
    if not updated:
        generated_by.append(
            {
                "Name": "dR2star",
                "Version": version,
                "Description": "dR2star processing using dr2",
            }
        )
    description["GeneratedBy"] = generated_by
    desc_path.write_text(json.dumps(description, indent=2, sort_keys=True) + "\n")
    return desc_path


def postprocess_dr2_json(
    json_path: Path,
    input_dir: Path,
    output_dir: Path,
    confounds_path: Path | list[Path],
    fd_thres: float | None,
    dvars_thresh: float | None,
) -> None:
    """Normalize paths in a dr2 JSON and add additional metadata."""
    data = json.loads(json_path.read_text())
    replacements = {
        str(input_dir): "bids:preprocessed:",
        str(output_dir): "bids::",
    }
    skip_keys = {"cmd", "collapse_cmd", "roistats_cmds", "volume_norm_cmds"}

    def _rewrite_str(value: str) -> str:
        if value.startswith("bids:"):
            return value
        for src, dst in replacements.items():
            src_path = Path(src)
            try:
                rel = Path(value).relative_to(src_path)
            except ValueError:
                continue
            return f"{dst}{rel.as_posix()}"
        return value

    def _rewrite(value, key: str | None = None):
        if key in skip_keys:
            return value
        if isinstance(value, str):
            return _rewrite_str(value)
        if isinstance(value, list):
            return [_rewrite(item) for item in value]
        if isinstance(value, dict):
            return {k: _rewrite(v, k) for k, v in value.items()}
        return value

    data = _rewrite(data)
    data.pop("censor_files", None)
    if isinstance(confounds_path, list):
        confounds_value = [_rewrite(str(path)) for path in confounds_path]
    else:
        confounds_value = _rewrite(str(confounds_path))
    payload = {
        "dr2star_generated": data,
        "confounds_file": confounds_value,
        "fd_thres": fd_thres,
        "dvars_thresh": dvars_thresh,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

def confounds_to_censor_file(
    confounds_tsv: str,
    censor_output_path: str,
    fd_thres: float = 0.3,
    dvars_thresh: float | None = None,
) -> np.ndarray:
    """Generate a censor file from fmriprep confounds.

    Parameters
    ----------
    confounds_tsv : str
        Path to fmriprep confounds TSV file.
    censor_output_path : str
        Path to write the censor file (one 0/1 per row).
    fd_thres : float, optional
        Framewise displacement threshold for censoring, by default 0.3.
    dvars_thresh : float | None, optional
        DVARS threshold for censoring, by default None (not used).

    """
    confounds = pd.read_csv(confounds_tsv, sep="\t")

    fd = confounds.get("framewise_displacement")
    if fd is None:
        raise ValueError("Framewise displacement column not found in confounds.")

    censor = np.ones(len(confounds), dtype=int)
    censor[fd > fd_thres] = 0

    if dvars_thresh is not None:
        dvars = confounds.get("dvars")
        if dvars is None:
            raise ValueError("DVARS column not found in confounds.")
        censor[dvars > dvars_thresh] = 0

    # Save the censor file that can be used with AFNI commands
    np.savetxt(censor_output_path, censor, fmt="%d")

    return


def group_confounds_by_entities(
    entities: list[str],
    confound_paths: list[Path | str],
) -> tuple[list[int], int, list[str]]:
    """Return aligned group IDs, group count, and reduced names after entity removal."""
    group_ids: list[int] = []
    reduced_names: list[str] = []
    key_to_id: dict[str, int] = {}
    for path in confound_paths:
        name = Path(path).name
        reduced_name = name
        for entity in entities:
            reduced_name = re.sub(rf"_{re.escape(entity)}-[^_]+", "", reduced_name)
        if reduced_name not in key_to_id:
            key_to_id[reduced_name] = len(key_to_id)
        group_ids.append(key_to_id[reduced_name])
        reduced_names.append(reduced_name)
    return group_ids, len(key_to_id), reduced_names


def merge_selected_volumes(
    volumes_by_path: dict[Path | str, list[int]],
    output_path: Path | str,
    needs_resampling: bool,
) -> Path:
    """Merge selected volumes into one NIfTI, resampling to the largest selection."""
    import nibabel as nib
    from nibabel import processing

    if not volumes_by_path:
        raise ValueError("volumes_by_path is empty.")

    selected_imgs: list[nib.Nifti1Image] = []
    selected_counts: list[int] = []
    for path, keep_mask in volumes_by_path.items():
        img_path = Path(path)
        img = nib.load(str(img_path), mmap=True)
        nvols = img.shape[3] if img.ndim > 3 else 1
        if len(keep_mask) != nvols:
            raise ValueError(
                f"Expected {nvols} mask entries for {img_path}, got {len(keep_mask)}."
            )
        keep_idxs = [i for i, flag in enumerate(keep_mask) if int(flag) == 1]
        if not keep_idxs:
            continue
        if img.ndim == 3:
            if keep_idxs != [0]:
                raise ValueError(
                    f"Invalid mask for 3D image {img_path}: {keep_idxs}."
                )
            subset_img = img
        else:
            keep_idxs = sorted(keep_idxs)
            if len(keep_idxs) == 1:
                subset_img = img.slicer[..., keep_idxs[0] : keep_idxs[0] + 1]
            elif keep_idxs == list(range(keep_idxs[0], keep_idxs[-1] + 1)):
                subset_img = img.slicer[..., keep_idxs[0] : keep_idxs[-1] + 1]
            else:
                subset_img = nib.funcs.concat_images(
                    [img.slicer[..., idx : idx + 1] for idx in keep_idxs],
                    axis=3,
                )
        selected_imgs.append(subset_img)
        selected_counts.append(len(keep_idxs))

    if not selected_imgs:
        raise ValueError("No volumes selected from any input image.")

    ref_idx = int(np.argmax(selected_counts))
    ref_img = selected_imgs[ref_idx]

    if needs_resampling:
        resampled_imgs: list[nib.Nifti1Image] = []
        for idx, temp_img in enumerate(selected_imgs):
            if idx == ref_idx:
                resampled_imgs.append(temp_img)
            else:
                resampled_imgs.append(processing.resample_from_to(temp_img, ref_img))
        selected_imgs = resampled_imgs

    if selected_imgs[0].ndim == 3:
        selected_imgs = [
            nib.Nifti1Image(
                np.expand_dims(np.asanyarray(img.dataobj), axis=3),
                img.affine,
                img.header,
            )
            for img in selected_imgs
        ]
    merged_img = nib.funcs.concat_images(selected_imgs, axis=3)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(merged_img, str(output_path))
    return output_path


def build_volume_selection_from_confounds(
    confound_paths: list[Path | str],
    nifti_paths: list[Path | str],
    fd_thres: float,
    dvars_thresh: float | None,
    sample_method: str | None,
    maxvols: int | None,
) -> dict[Path, list[int]]:
    """Return per-NIfTI 0/1 selection masks after FD/DVARS and group sampling."""
    if len(confound_paths) != len(nifti_paths):
        raise ValueError(
            "confound_paths and nifti_paths must have the same number of entries."
        )
    if sample_method is None:
        sample_method = "first"
    if sample_method not in {"first", "last", "random"}:
        raise ValueError(f"Unsupported sample_method '{sample_method}'.")

    per_run_keep: list[list[int]] = []
    per_run_nvols: list[int] = []

    for confound_path in confound_paths:
        confound_path = Path(confound_path)
        confounds = pd.read_csv(confound_path, sep="\t")
        fd = confounds.get("framewise_displacement")
        if fd is None:
            raise ValueError(
                f"Framewise displacement column not found in {confound_path}."
            )

        censor = np.ones(len(confounds), dtype=int)
        censor[fd > fd_thres] = 0

        if dvars_thresh is not None:
            dvars = confounds.get("dvars")
            if dvars is None:
                raise ValueError(f"DVARS column not found in {confound_path}.")
            censor[dvars > dvars_thresh] = 0

        keep_indices = np.where(censor == 1)[0].tolist()
        per_run_keep.append(keep_indices)
        per_run_nvols.append(len(censor))

    global_indices: list[tuple[int, int]] = []
    for run_idx, keep_indices in enumerate(per_run_keep):
        for vol_idx in keep_indices:
            global_indices.append((run_idx, vol_idx))

    if maxvols is not None and maxvols > 0:
        if sample_method == "first":
            selected = global_indices[:maxvols]
        elif sample_method == "last":
            selected = global_indices[-maxvols:]
        else:
            rng = np.random.default_rng()
            if maxvols >= len(global_indices):
                selected = global_indices
            else:
                choice_idx = rng.choice(len(global_indices), size=maxvols, replace=False)
                selected = [global_indices[idx] for idx in choice_idx]
    else:
        selected = global_indices

    selected_by_run: dict[int, set[int]] = {}
    for run_idx, vol_idx in selected:
        selected_by_run.setdefault(run_idx, set()).add(vol_idx)

    selections: dict[Path, list[int]] = {}
    for run_idx, nifti_path in enumerate(nifti_paths):
        nvols = per_run_nvols[run_idx]
        mask = np.zeros(nvols, dtype=int)
        for vol_idx in selected_by_run.get(run_idx, set()):
            mask[vol_idx] = 1
        selections[Path(nifti_path)] = mask.tolist()

    return selections


def average_dR2star_vols(
    entities: list[str],
    anat_dir: Path,
    output_dir: Path,
) -> dict[str, list[str]]:
    """Average dR2star volumes across selected BIDS entities."""
    reduced_map: dict[str, list[str]] = {}
    for path in sorted(anat_dir.glob("*dR2starmap.nii.gz")):
        name = path.name
        base = name[: -len(".nii.gz")]
        reduced_base = base
        for entity in entities:
            reduced_base = re.sub(rf"_{re.escape(entity)}-[^_]+", "", reduced_base)
        reduced_name = f"{reduced_base}.nii.gz"
        reduced_map.setdefault(reduced_name, []).append(name)

    import nibabel as nib
    from nibabel import processing

    for output_vol_name, input_vols in reduced_map.items():
        if len(input_vols) == 1:
            print(
                "Only one volume found for the following grouping, "
                f"skipping averaging: {output_vol_name}"
            )
            continue
        else:
            print(f"Found {len(input_vols)} volumes to average for {output_vol_name}")
            print(f"Input volumes: {input_vols}")
            print(f"Output volume (to be created): {output_vol_name}")

            imgs = []
            num_frames = np.zeros(len(input_vols), dtype=int)
            for idx, vol_name in enumerate(input_vols):
                vol_path = anat_dir / vol_name

                img = nib.load(str(vol_path))
                imgs.append(img)
                json_path = Path(str(vol_path).replace(".nii.gz", ".json"))
                with open(json_path, "r") as f:
                    metadata = json.load(f)
                concat_nvol = metadata.get("concat_nvol")
                if concat_nvol is None:
                    concat_nvol = metadata.get("dr2star_generated", {}).get("concat_nvol")
                if concat_nvol is None:
                    raise KeyError(
                        f"Metadata key 'concat_nvol' not found in {json_path}"
                    )
                num_frames[idx] = concat_nvol
            total_frames = np.sum(num_frames)
            best_image = imgs[np.argmax(num_frames)]
            relative_weighting = (num_frames / total_frames).tolist()
            source_data = []
            for vol_name in input_vols:
                vol_path = anat_dir / vol_name
                bids_uri = str(vol_path).replace(str(output_dir), "bids::")
                source_data.append(bids_uri)

            if ('space-T1w' in output_vol_name) or ('space-T2w' in output_vol_name):
                best_zooms = np.array(best_image.header.get_zooms()[:3])
                voxel_mismatch = []
                for vol_name, temp_img in zip(input_vols, imgs):
                    temp_zooms = np.array(temp_img.header.get_zooms()[:3])
                    if not np.allclose(temp_zooms, best_zooms):
                        voxel_mismatch.append((vol_name, temp_zooms))
                if voxel_mismatch:
                    print(
                        "WARNING: resampling volumes with differing voxel sizes; "
                        f"using reference zooms {best_zooms.tolist()} for {output_vol_name}."
                    )
                    for vol_name, temp_zooms in voxel_mismatch:
                        print(
                            f"  - {vol_name}: zooms={temp_zooms.tolist()}"
                        )
                resampled_imgs = []
                for temp_img in imgs:
                    if temp_img == best_image:
                        resampled_imgs.append(temp_img)
                    else:
                        resampled_imgs.append(processing.resample_from_to(temp_img, best_image))
                concat_data = np.zeros(best_image.shape)
                total_frames = np.sum(num_frames)
                for i, temp_img in enumerate(resampled_imgs):
                    concat_data += temp_img.get_fdata()*num_frames[i]/total_frames
                concat_img = nib.Nifti1Image(concat_data, best_image.affine, best_image.header)
                output_path = anat_dir / output_vol_name
                nib.save(concat_img, str(output_path))
            elif ('space-MNI' in output_vol_name):
                concat_data = np.zeros(best_image.shape)
                total_frames = np.sum(num_frames)
                for i, temp_img in enumerate(imgs):
                    concat_data += temp_img.get_fdata()*num_frames[i]/total_frames
                concat_img = nib.Nifti1Image(concat_data, best_image.affine, best_image.header)
                output_path = anat_dir / output_vol_name
                nib.save(concat_img, str(output_path))
            else:
                raise ValueError(f"Unexpected space entity in output volume name: {output_vol_name}")

            json_path = Path(str(output_path).replace(".nii.gz", ".json"))
            json_path.write_text(
                json.dumps(
                    {
                        "source_data": source_data,
                        "relative_weighting": relative_weighting,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )

    return
