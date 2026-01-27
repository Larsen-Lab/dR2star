from __future__ import annotations

import sys
from pathlib import Path

project = "dR2star"
author = "dR2star contributors"
release = "dev"

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

extensions = ["sphinxarg.ext"]

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = []
