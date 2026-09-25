from pathlib import Path
import sys
import uuid
import warnings

import nbformat
from nbformat.warnings import MissingIDFieldWarning


def normalize_notebooks(root: Path) -> int:
    warnings.filterwarnings("ignore", category=MissingIDFieldWarning)
    updated = 0

    for notebook_path in sorted(root.glob("*.ipynb")):
        nb = nbformat.read(notebook_path, as_version=4)
        for cell in nb.cells:
            if "id" not in cell:
                cell["id"] = uuid.uuid4().hex[:8]

            if cell.get("cell_type") == "code":
                cell.setdefault("metadata", {})["trusted"] = True

        # Always write back: nbformat may auto-fill missing IDs during read.
        nbformat.write(nb, notebook_path)
        print(f"[normalize] updated {notebook_path}")
        updated += 1

    return updated


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    normalize_notebooks(target)
