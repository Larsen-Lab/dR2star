"""Generate CLI usage docs from dR2star.my_parser."""

from __future__ import annotations

import sys
from pathlib import Path

import mkdocs_gen_files


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from dR2star.my_parser import get_parser

    parser = get_parser()
    usage_text = parser.format_help()

    with mkdocs_gen_files.open("usage.md", "w") as handle:
        handle.write("# Usage\n\n")
        handle.write("```\n")
        handle.write(usage_text)
        if not usage_text.endswith("\n"):
            handle.write("\n")
        handle.write("```\n")


if __name__ == "__main__":
    main()
