"""Execute the notebook using hwenv and project-local dependencies."""
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
PACKAGES = BASE / ".packages"
sys.path.insert(0, str(PACKAGES))
os.environ["PYTHONPATH"] = str(PACKAGES)
os.environ["JUPYTER_PATH"] = str(PACKAGES / "share" / "jupyter")

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def main():
    path = next(BASE.glob("*.ipynb"))
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"
    ]
    client = NotebookClient(
        notebook, km=manager, timeout=600,
        resources={"metadata": {"path": str(BASE)}},
    )
    client.on_cell_start = lambda cell, cell_index, **kwargs: print(
        f"Cell {cell_index + 1}/{len(notebook.cells)}", flush=True
    )
    client.execute()
    nbformat.write(notebook, path)
    print("Saved executed notebook:", path.name)
    for name in ("q1_summary.csv", "q2_summary.csv"):
        print((BASE / "results" / name).read_text())


if __name__ == "__main__":
    main()
