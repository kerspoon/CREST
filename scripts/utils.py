"""Utility functions for CREST validation scripts."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


def get_next_run_number(base_dir: Path, prefix: str, date: str) -> int:
    """
    Scan directory for existing runs with the given prefix and date.

    Args:
        base_dir: Directory to scan (e.g., output/monte_carlo/)
        prefix: Run prefix (e.g., "python_1000runs")
        date: Date string in YYYYMMDD format

    Returns:
        Next available run number (1-based)

    Example:
        If directory contains:
            python_1000runs_20250113_01/
            python_1000runs_20250113_02/
        Returns: 3
    """
    if not base_dir.exists():
        return 1

    pattern = re.compile(rf"^{re.escape(prefix)}_{re.escape(date)}_(\d+)$")
    max_num = 0

    for item in base_dir.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                num = int(match.group(1))
                max_num = max(max_num, num)

    return max_num + 1


def create_output_dir(
    run_type: str,
    base_dir: str = "output",
    prefix: str = "",
    date: Optional[str] = None
) -> Path:
    """
    Create auto-numbered output directory.

    Args:
        run_type: Type of run (e.g., "monte_carlo", "rng_validation")
        base_dir: Base output directory (default: "output")
        prefix: Run prefix (e.g., "python_1000runs")
        date: Date string in YYYYMMDD format (default: today)

    Returns:
        Path to created directory

    Example:
        create_output_dir("monte_carlo", prefix="python_1000runs")
        → output/monte_carlo/python_1000runs_20250113_01/
    """
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    base_path = Path(base_dir) / run_type
    base_path.mkdir(parents=True, exist_ok=True)

    run_num = get_next_run_number(base_path, prefix, date)
    output_path = base_path / f"{prefix}_{date}_{run_num:02d}"
    output_path.mkdir(parents=True, exist_ok=True)

    return output_path


def parse_run_dir(run_dir: str) -> Tuple[str, str]:
    """
    Parse run directory name to extract date and run number.

    Args:
        run_dir: Directory name or path

    Returns:
        Tuple of (date, run_num) as strings

    Example:
        parse_run_dir("python_1000runs_20250113_01")
        → ("20250113", "01")
    """
    path = Path(run_dir)
    name = path.name

    # Match pattern: prefix_YYYYMMDD_NN
    match = re.search(r"_(\d{8})_(\d{2})$", name)
    if not match:
        raise ValueError(f"Invalid run directory name: {name}")

    return match.group(1), match.group(2)


def create_validation_dir(
    python_dir: str,
    excel_dir: str,
    run_type: str,
    base_dir: str = "output"
) -> Path:
    """
    Create validation directory linking Python and Excel runs.

    Args:
        python_dir: Path to Python run directory
        excel_dir: Path to Excel run directory
        run_type: Type of validation (e.g., "monte_carlo", "rng_validation")
        base_dir: Base output directory (default: "output")

    Returns:
        Path to created validation directory

    Example:
        create_validation_dir(
            "output/monte_carlo/python_1000runs_20250113_01",
            "output/monte_carlo/excel_20runs_20250113_01",
            "monte_carlo"
        )
        → output/monte_carlo/validation_p20250113_01_e20250113_01/
    """
    p_date, p_num = parse_run_dir(python_dir)
    e_date, e_num = parse_run_dir(excel_dir)

    validation_name = f"validation_p{p_date}_{p_num}_e{e_date}_{e_num}"
    validation_path = Path(base_dir) / run_type / validation_name
    validation_path.mkdir(parents=True, exist_ok=True)

    return validation_path


def save_metadata(
    validation_dir: Path,
    python_dir: str,
    excel_dir: str,
    **kwargs
) -> None:
    """
    Save metadata.json with source run info and optional additional fields.

    Args:
        validation_dir: Path to validation directory
        python_dir: Path to Python run directory
        excel_dir: Path to Excel run directory
        **kwargs: Additional metadata fields to save

    Example:
        save_metadata(
            validation_path,
            "output/monte_carlo/python_1000runs_20250113_01",
            "output/monte_carlo/excel_20runs_20250113_01",
            iterations_python=1000,
            iterations_excel=20
        )
    """
    metadata = {
        "python_run": str(Path(python_dir).absolute()),
        "excel_run": str(Path(excel_dir).absolute()),
        "validation_date": datetime.now().isoformat(),
        **kwargs
    }

    metadata_file = validation_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def get_project_root() -> Path:
    """
    Get the project root directory (where this script is located).

    Returns:
        Path to project root
    """
    # scripts/ is one level below project root
    return Path(__file__).parent.parent


def get_python_main() -> Path:
    """
    Get path to python/main.py

    Returns:
        Path to main.py
    """
    return get_project_root() / "python" / "main.py"
