"""
Wildfire Risk Classifier — End-to-End Pipeline Runner

Usage:
    python run_pipeline.py              # full pipeline
    python run_pipeline.py --skip-install   # skip pip install
    python run_pipeline.py --only ml    # only the ML notebooks
    python run_pipeline.py --only data  # only the data processing notebooks
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent

DATA_PROCESSING_NOTEBOOKS = [
    ROOT / "src" / "data_processing" / "CensusTract_SVI.ipynb",
    ROOT / "src" / "data_processing" / "Escape_Route.ipynb",
    ROOT / "src" / "data_processing" / "FireHydrants.ipynb",
    ROOT / "src" / "data_processing" / "FireStations.ipynb",
    ROOT / "src" / "data_processing" / "Historic_Wildfire.ipynb",
    ROOT / "src" / "data_processing" / "Hospital.ipynb",
]

ML_PIPELINE_NOTEBOOKS = [
    ROOT / "notebooks" / "exploration.ipynb",
    ROOT / "notebooks" / "feature_engineering.ipynb",
    ROOT / "notebooks" / "model_trainning.ipynb",
]


def install_requirements():
    req_file = ROOT / "requirements.txt"
    print(f"\n{'='*60}")
    print("Step 0: Installing dependencies")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
        check=True,
    )
    return result.returncode == 0


def run_notebook(notebook_path: Path) -> bool:
    print(f"\n  Running: {notebook_path.relative_to(ROOT)}")
    start = time.time()
    result = subprocess.run(
        [
            sys.executable, "-m", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=600",
            "--ExecutePreprocessor.kernel_name=python3",
            str(notebook_path),
        ],
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"  FAILED ({elapsed:.1f}s)")
        print(result.stderr[-2000:] if result.stderr else "(no stderr)")
        return False
    print(f"  OK ({elapsed:.1f}s)")
    return True


def run_stage(label: str, notebooks: list[Path]) -> bool:
    print(f"\n{'='*60}")
    print(f"Stage: {label}")
    print(f"{'='*60}")
    for nb in notebooks:
        if not nb.exists():
            print(f"  SKIP (not found): {nb.relative_to(ROOT)}")
            continue
        if not run_notebook(nb):
            print(f"\nPipeline aborted at: {nb.name}")
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Run the wildfire ML pipeline.")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip install")
    parser.add_argument(
        "--only",
        choices=["data", "ml"],
        help="Run only the data processing or ML notebooks",
    )
    args = parser.parse_args()

    print("\nWildfire Risk Classifier — Pipeline Runner")

    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "cache").mkdir(parents=True, exist_ok=True)

    if not args.skip_install:
        install_requirements()

    run_data = args.only in (None, "data")
    run_ml = args.only in (None, "ml")

    if run_data:
        ok = run_stage("Data Processing (src/data_processing)", DATA_PROCESSING_NOTEBOOKS)
        if not ok:
            sys.exit(1)

    if run_ml:
        ok = run_stage("ML Pipeline (notebooks)", ML_PIPELINE_NOTEBOOKS)
        if not ok:
            sys.exit(1)

    print(f"\n{'='*60}")
    print("Pipeline completed successfully.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
