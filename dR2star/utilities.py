#!/usr/bin/env python3
import json
import os
from pathlib import Path

import pandas as pd
import numpy as np


def _normalize_labels(labels: list[str], prefix: str) -> list[str]:
    return [label.removeprefix(prefix) for label in labels]


def _discover_subjects(input_dir: Path, requested: list[str]) -> list[str]:
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
    if filename.endswith("_desc-confounds_timeseries.tsv"):
        return filename.replace("_desc-confounds_timeseries.tsv", suffix)
    if filename.endswith("_desc-confounds_regressors.tsv"):
        return filename.replace("_desc-confounds_regressors.tsv", suffix)
    raise NameError(f"Unexpected confound file name format: {filename}")


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
            if entry.get("Description") == "dr2star processing using tat2":
                entry["Description"] = "dR2star processing using tat2"
            else:
                entry.setdefault("Description", "dR2star processing using tat2")
            updated = True
            break
    if not updated:
        generated_by.append(
            {
                "Name": "dR2star",
                "Version": version,
                "Description": "dR2star processing using tat2",
            }
        )
    description["GeneratedBy"] = generated_by
    desc_path.write_text(json.dumps(description, indent=2, sort_keys=True) + "\n")
    return desc_path


def postprocess_tat2_json(
    json_path: Path,
    input_dir: Path,
    output_dir: Path,
    confounds_path: Path,
    fd_thres: float | None,
    dvars_thresh: float | None,
) -> None:
    """Normalize paths in a tat2 JSON and add additional metadata."""
    data = json.loads(json_path.read_text())
    replacements = {
        str(input_dir): "bids:preprocessed:",
        str(output_dir): "bids::",
    }

    def _rewrite(value):
        if isinstance(value, str):
            for src, dst in replacements.items():
                value = value.replace(src, dst)
            return value
        if isinstance(value, list):
            return [_rewrite(item) for item in value]
        if isinstance(value, dict):
            return {key: _rewrite(item) for key, item in value.items()}
        return value

    data = _rewrite(data)
    confounds_value = _rewrite(str(confounds_path))
    data["confounds_file"] = confounds_value
    data["fd_thres"] = fd_thres
    data["dvars_thresh"] = dvars_thresh
    json_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

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


def concat_dR2star_vols(entities: list[str], anat_dir: Path) -> None:
    """Placeholder for concatenating dR2star volumes."""
    print(
        "The concat_dR2star_vols function is not implemented yet. "
        "Volumes will not be concatenated until implementation is finished."
    )
