"""
SLURM script generation for cluster-based radiomics extraction.

Supports three strategies:
  - single: one SLURM job processes the entire cohort (your current approach)
  - array:  one SLURM array task per subject (max parallelism)
  - chunked: split cohort into N chunks, one job per chunk (balanced)

Usage:
    >>> from radiomicviz._slurm import generate_slurm_scripts
    >>> scripts = generate_slurm_scripts(
    ...     subjects_csv="cohort.csv",
    ...     image_col="t1_path",
    ...     mask_col="mask_path",
    ...     strategy="array",
    ...     conda_env="radiomics_env",
    ... )
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional, Union

import pandas as pd

logger = logging.getLogger("radiomicviz.slurm")

# ---------------------------------------------------------------------------
# SLURM header template
# ---------------------------------------------------------------------------
_SLURM_HEADER = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/{job_name}_%j{array_idx_suffix}.log
#SBATCH --error={log_dir}/{job_name}_%j{array_idx_suffix}.err
#SBATCH --time={time_limit}
#SBATCH --constraint={constraint}
{partition_line}{constraint_line}{array_line}
set -euo pipefail

echo "============================================="
echo "RadiomicViz SLURM Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "============================================="
"""

_CONDA_BLOCK = """
# --- Conda setup ---
source {conda_sh}
conda activate {conda_env}
export PATH="$(conda info --base)/envs/{conda_env}/bin:$PATH"
echo "Python: $(which python) ($(python --version))"
echo "radiomicviz: $(python -c 'import radiomicviz; print(radiomicviz.__version__)' 2>/dev/null || echo 'not installed')"
echo ""
"""


def generate_slurm_scripts(
    subjects_csv: Union[str, Path],
    image_col: str,
    mask_col: str,
    *,
    preset: str = "mri-default",
    config: Optional[str] = None,
    output_dir: str = "./radiomics_output/",
    mode: str = "roi",
    label: Optional[int] = None,
    label_col: Optional[str] = None,
    subject_id_col: Optional[str] = None,
    modality: Optional[str] = None,
    n_jobs: int = 1,
    strategy: str = "single",
    chunks: int = 10,
    partition: Optional[str] = None,
    constraint: Optional[str] = None,
    time_limit: str = "04:00:00",
    mem: str = "16G",
    conda_env: Optional[str] = None,
    conda_sh: Optional[str] = None,
    script_dir: str = "./slurm_scripts/",
) -> list[Path]:
    """
    Generate SLURM submission scripts for radiomics extraction.

    Parameters
    ----------
    subjects_csv : str or Path
        Path to the subjects CSV file.
    image_col, mask_col : str
        Column names for image and mask paths.
    preset : str
        Preset name.
    config : str, optional
        Path to custom config (overrides preset).
    output_dir : str
        Where extraction results are saved.
    mode : str
        "roi" or "voxelwise".
    label : int, optional
        Specific mask label.
    label_col : str, optional
        Column with per-subject labels.
    subject_id_col : str, optional
        Column with subject IDs.
    modality : str, optional
        Modality label.
    n_jobs : int
        Parallel workers per SLURM job.
    strategy : str
        "single", "array", or "chunked".
    chunks : int
        Number of chunks for "chunked" strategy.
    partition : str, optional
        SLURM partition.
    constraint : str, optional
        SLURM constraint string.
    time_limit : str
        SLURM time limit (HH:MM:SS).
    mem : str
        Memory per job.
    conda_env : str, optional
        Conda environment name.
    conda_sh : str, optional
        Path to conda.sh. Auto-detected if not provided.
    script_dir : str
        Directory for generated scripts.

    Returns
    -------
    list of Path
        Paths to the generated SLURM scripts.
    """
    subjects_csv = Path(subjects_csv).resolve()
    script_dir = Path(script_dir)
    script_dir.mkdir(parents=True, exist_ok=True)

    log_dir = script_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # Auto-detect conda.sh if needed
    if conda_env and not conda_sh:
        conda_sh = _guess_conda_sh()

    # Build the radiomicviz CLI command
    base_cmd = _build_base_cmd(
        image_col=image_col, mask_col=mask_col, preset=preset,
        config=config, output_dir=output_dir, mode=mode, label=label,
        label_col=label_col, subject_id_col=subject_id_col,
        modality=modality, n_jobs=n_jobs,
    )

    if strategy == "single":
        return _gen_single(
            subjects_csv, base_cmd, script_dir, log_dir,
            partition, constraint, time_limit, mem,
            conda_env, conda_sh,
        )
    elif strategy == "array":
        return _gen_array(
            subjects_csv, image_col, mask_col, base_cmd,
            script_dir, log_dir,
            partition, constraint, time_limit, mem,
            conda_env, conda_sh, preset, config, output_dir,
            mode, label, modality,
        )
    elif strategy == "chunked":
        return _gen_chunked(
            subjects_csv, base_cmd, chunks, script_dir, log_dir,
            partition, constraint, time_limit, mem,
            conda_env, conda_sh, image_col, mask_col,
        )
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'single', 'array', or 'chunked'.")


# ---------------------------------------------------------------------------
# Strategy: single (one big job)
# ---------------------------------------------------------------------------
def _gen_single(
    subjects_csv, base_cmd, script_dir, log_dir,
    partition, constraint, time_limit, mem,
    conda_env, conda_sh,
) -> list[Path]:
    header = _make_header(
        "radiomicviz_batch", log_dir, partition, constraint,
        time_limit, mem,
    )
    conda = _make_conda_block(conda_env, conda_sh)

    script = f"""{header}{conda}
# --- Run batch extraction ---
radiomicviz batch-extract \\
    --subjects {subjects_csv} \\
    {base_cmd}

echo ""
echo "============================================="
echo "Job complete: $(date)"
echo "============================================="
"""
    path = script_dir / "submit_batch.sh"
    path.write_text(script)
    path.chmod(0o755)
    return [path]


# ---------------------------------------------------------------------------
# Strategy: array (one task per subject)
# ---------------------------------------------------------------------------
def _gen_array(
    subjects_csv, image_col, mask_col, base_cmd,
    script_dir, log_dir,
    partition, constraint, time_limit, mem,
    conda_env, conda_sh, preset, config, output_dir,
    mode, label, modality,
) -> list[Path]:
    df = pd.read_csv(subjects_csv)
    n_subjects = len(df)

    header = _make_header(
        "radiomicviz_array", log_dir, partition, constraint,
        time_limit, mem, array_size=n_subjects,
    )
    conda = _make_conda_block(conda_env, conda_sh)

    # Build per-subject extract command (not batch)
    preset_flag = f"--preset {preset}" if preset and not config else ""
    config_flag = f"--config {config}" if config else ""
    mode_flag = f"--mode {mode}"
    label_flag = f"--label {label}" if label else ""
    modality_flag = f"--modality {modality}" if modality else ""

    script = f"""{header}{conda}
# --- Per-subject extraction via SLURM array ---
SUBJECTS_CSV="{subjects_csv}"
IDX=$SLURM_ARRAY_TASK_ID  # 0-indexed

# Extract image and mask paths from CSV row
IMAGE=$(python -c "import pandas as pd; df=pd.read_csv('$SUBJECTS_CSV'); print(df.iloc[$IDX]['{image_col}'])")
MASK=$(python -c "import pandas as pd; df=pd.read_csv('$SUBJECTS_CSV'); print(df.iloc[$IDX]['{mask_col}'])")

echo "Processing subject index $IDX"
echo "  Image: $IMAGE"
echo "  Mask:  $MASK"

radiomicviz extract \\
    --image "$IMAGE" \\
    --mask "$MASK" \\
    {preset_flag} {config_flag} {mode_flag} {label_flag} {modality_flag} \\
    --output {output_dir}/subject_${{IDX}}_features.csv

echo "Subject $IDX complete: $(date)"
"""
    path = script_dir / "submit_array.sh"
    path.write_text(script)
    path.chmod(0o755)

    # Also generate a merge script
    merge_script = f"""#!/bin/bash
# Run this AFTER all array jobs complete to merge per-subject results
echo "Merging results from {output_dir}..."
python -c "
import pandas as pd
from pathlib import Path
csvs = sorted(Path('{output_dir}').glob('subject_*_features.csv'))
dfs = [pd.read_csv(c) for c in csvs]
combined = pd.concat(dfs, ignore_index=True)
combined.to_csv('{output_dir}/combined_features.csv', index=False)
print(f'Merged {{len(csvs)}} files → {output_dir}/combined_features.csv')
"
"""
    merge_path = script_dir / "merge_results.sh"
    merge_path.write_text(merge_script)
    merge_path.chmod(0o755)

    return [path, merge_path]


# ---------------------------------------------------------------------------
# Strategy: chunked (N jobs, each processing a subset)
# ---------------------------------------------------------------------------
def _gen_chunked(
    subjects_csv, base_cmd, chunks, script_dir, log_dir,
    partition, constraint, time_limit, mem,
    conda_env, conda_sh, image_col, mask_col,
) -> list[Path]:
    df = pd.read_csv(subjects_csv)
    n_subjects = len(df)
    chunk_size = math.ceil(n_subjects / chunks)

    # Split CSV into chunks
    chunks_dir = script_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    chunk_csvs = []
    for i in range(chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_subjects)
        if start >= n_subjects:
            break
        chunk_df = df.iloc[start:end]
        chunk_path = chunks_dir / f"chunk_{i:03d}.csv"
        chunk_df.to_csv(chunk_path, index=False)
        chunk_csvs.append(chunk_path)

    # Generate one submit-all script
    header = _make_header(
        "radiomicviz_chunked", log_dir, partition, constraint,
        time_limit, mem, array_size=len(chunk_csvs),
    )
    conda = _make_conda_block(conda_env, conda_sh)

    script = f"""{header}{conda}
# --- Chunked batch extraction ---
CHUNK_DIR="{chunks_dir.resolve()}"
CHUNK_IDX=$SLURM_ARRAY_TASK_ID
CHUNK_CSV="${{CHUNK_DIR}}/chunk_$(printf '%03d' $CHUNK_IDX).csv"

echo "Processing chunk $CHUNK_IDX: $CHUNK_CSV"

radiomicviz batch-extract \\
    --subjects "$CHUNK_CSV" \\
    {base_cmd} \\
    -o ./radiomics_output/chunk_$CHUNK_IDX/

echo "Chunk $CHUNK_IDX complete: $(date)"
"""
    path = script_dir / "submit_chunked.sh"
    path.write_text(script)
    path.chmod(0o755)

    return [path]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_base_cmd(**kwargs) -> str:
    """Build the CLI flags string from kwargs."""
    parts = []
    mapping = {
        "image_col": "--image-col",
        "mask_col": "--mask-col",
        "preset": "--preset",
        "config": "--config",
        "output_dir": "-o",
        "mode": "--mode",
        "label": "--label",
        "label_col": "--label-col",
        "subject_id_col": "--subject-id-col",
        "modality": "--modality",
        "n_jobs": "-n",
    }
    for key, flag in mapping.items():
        val = kwargs.get(key)
        if val is not None:
            parts.append(f"{flag} {val}")
    return " \\\n    ".join(parts)


def _make_header(
    job_name, log_dir, partition, constraint, time_limit, mem,
    array_size=None,
) -> str:
    partition_line = f"#SBATCH --partition={partition}\n" if partition else ""
    constraint_line = f'#SBATCH --constraint="{constraint}"\n' if constraint else ""
    array_line = f"#SBATCH --array=0-{array_size - 1}\n" if array_size else ""
    array_idx_suffix = "_%a" if array_size else ""

    return _SLURM_HEADER.format(
        job_name=job_name,
        log_dir=log_dir.resolve(),
        partition_line=partition_line,
        constraint_line=constraint_line,
        array_line=array_line,
        array_idx_suffix=array_idx_suffix,
        time_limit=time_limit,
        mem=mem,
    )


def _make_conda_block(conda_env, conda_sh) -> str:
    if not conda_env:
        return "\n# No conda environment specified\n"
    return _CONDA_BLOCK.format(conda_env=conda_env, conda_sh=conda_sh or "~/miniconda3/etc/profile.d/conda.sh")


def _guess_conda_sh() -> str:
    """Try to find conda.sh in common locations."""
    import os
    candidates = [
        os.path.expanduser("~/miniconda3/etc/profile.d/conda.sh"),
        os.path.expanduser("~/anaconda3/etc/profile.d/conda.sh"),
        os.path.expanduser("~/conda/etc/profile.d/conda.sh"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "~/miniconda3/etc/profile.d/conda.sh"  # fallback
