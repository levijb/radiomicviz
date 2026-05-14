"""
Command-line interface for RadiomicViz.

Usage:
    radiomicviz extract --image t1.nii.gz --mask mask.nii.gz --preset mri-default
    radiomicviz batch-extract --subjects cohort.csv --image-col t1_path --mask-col mask_path
    radiomicviz validate --image t1.nii.gz --mask mask.nii.gz
    radiomicviz show-preset mri-default
    radiomicviz list-presets
    radiomicviz generate-slurm --subjects cohort.csv --image-col t1 --mask-col mask
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from radiomicviz._version import __version__


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)-.19s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


@click.group()
@click.version_option(__version__, prog_name="radiomicviz")
def cli():
    """RadiomicViz: Interactive radiomics extraction and visualization."""
    pass


# -------------------------------------------------------------------------
# extract
# -------------------------------------------------------------------------
@cli.command()
@click.option("-i", "--image", required=True, type=click.Path(exists=True),
              help="Path to NIfTI image file")
@click.option("-m", "--mask", required=True, type=click.Path(exists=True),
              help="Path to NIfTI mask file")
@click.option("-p", "--preset", default=None, help="Built-in preset name")
@click.option("-c", "--config", default=None, type=click.Path(exists=True),
              help="Custom PyRadiomics YAML config")
@click.option("-o", "--output", default="features.csv",
              help="Output CSV path (default: features.csv)")
@click.option("--mode", type=click.Choice(["roi", "voxelwise"]), default="roi",
              help="Extraction mode")
@click.option("-l", "--label", type=int, default=None,
              help="Specific mask label to extract")
@click.option("--modality", default=None, help="Modality label (e.g. T1, FLAIR)")
@click.option("--subject-id", default=None, help="Subject identifier for metadata")
@click.option("--skip-validation", is_flag=True, help="Skip input validation")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def extract(image, mask, preset, config, output, mode, label, modality,
            subject_id, skip_validation, verbose):
    """Extract radiomic features from a single image-mask pair."""
    _setup_logging(verbose)

    from radiomicviz import extract as _extract

    try:
        result = _extract(
            image=image, mask=mask, preset=preset, config=config,
            mode=mode, label=label, modality=modality,
            subject_id=subject_id, skip_validation=skip_validation,
        )
        result.to_csv(output)
        click.echo(result.summary())
        click.echo(f"\nFeatures saved to {output}")
    except Exception as exc:
        click.secho(f"Error: {exc}", fg="red", err=True)
        raise SystemExit(1)


# -------------------------------------------------------------------------
# batch-extract
# -------------------------------------------------------------------------
@cli.command("batch-extract")
@click.option("-s", "--subjects", required=True, type=click.Path(exists=True),
              help="Subjects CSV file")
@click.option("--image-col", required=True, help="Column name for image paths")
@click.option("--mask-col", required=True, help="Column name for mask paths")
@click.option("-p", "--preset", default=None, help="Built-in preset name")
@click.option("-c", "--config", default=None, type=click.Path(exists=True),
              help="Custom config YAML")
@click.option("-o", "--output-dir", default="./radiomics_output/",
              help="Output directory")
@click.option("--mode", type=click.Choice(["roi", "voxelwise"]), default="roi")
@click.option("-l", "--label", type=int, default=None,
              help="Label to extract (overrides per-subject)")
@click.option("--label-col", default=None, help="Column with per-subject labels")
@click.option("--subject-id-col", default=None, help="Column with subject IDs")
@click.option("--modality", default=None, help="Modality label")
@click.option("-n", "--n-jobs", type=int, default=1, help="Parallel workers")
@click.option("--skip-validation", is_flag=True)
@click.option("-v", "--verbose", is_flag=True)
def batch_extract_cmd(subjects, image_col, mask_col, preset, config, output_dir,
                      mode, label, label_col, subject_id_col, modality, n_jobs,
                      skip_validation, verbose):
    """Extract radiomic features from a cohort of subjects."""
    _setup_logging(verbose)

    from radiomicviz import batch_extract

    try:
        results = batch_extract(
            subjects_csv=subjects, image_col=image_col, mask_col=mask_col,
            preset=preset, config=config, mode=mode, label=label,
            label_col=label_col, subject_id_col=subject_id_col,
            modality=modality, n_jobs=n_jobs, output_dir=output_dir,
            skip_validation=skip_validation,
        )
        click.echo(f"\nExtracted {len(results)} subjects → {output_dir}")
    except Exception as exc:
        click.secho(f"Error: {exc}", fg="red", err=True)
        raise SystemExit(1)


# -------------------------------------------------------------------------
# validate
# -------------------------------------------------------------------------
@cli.command()
@click.option("-i", "--image", required=True, type=click.Path(),
              help="Path to NIfTI image")
@click.option("-m", "--mask", required=True, type=click.Path(),
              help="Path to NIfTI mask")
@click.option("-l", "--label", type=int, default=None)
def validate(image, mask, label):
    """Validate an image-mask pair before extraction."""
    from radiomicviz import validate_inputs

    report = validate_inputs(image, mask, label=label)
    click.echo(str(report))
    if not report.ok:
        raise SystemExit(1)


# -------------------------------------------------------------------------
# Preset management
# -------------------------------------------------------------------------
@cli.command("list-presets")
def list_presets_cmd():
    """List all available built-in presets."""
    from radiomicviz import list_presets

    presets = list_presets()
    click.echo("Available presets:")
    for p in presets:
        click.echo(f"  • {p}")


@cli.command("show-preset")
@click.argument("name")
def show_preset_cmd(name):
    """Show the YAML content of a built-in preset."""
    from radiomicviz import show_preset

    try:
        show_preset(name)
    except FileNotFoundError as exc:
        click.secho(str(exc), fg="red", err=True)
        raise SystemExit(1)


# -------------------------------------------------------------------------
# SLURM generation
# -------------------------------------------------------------------------
@cli.command("generate-slurm")
@click.option("-s", "--subjects", required=True, type=click.Path(exists=True),
              help="Subjects CSV file")
@click.option("--image-col", required=True, help="Column for image paths")
@click.option("--mask-col", required=True, help="Column for mask paths")
@click.option("-p", "--preset", default="mri-default", help="Preset name")
@click.option("-c", "--config", default=None, type=click.Path(),
              help="Custom config YAML (overrides preset)")
@click.option("-o", "--output-dir", default="./radiomics_output/",
              help="Output directory for results")
@click.option("--mode", type=click.Choice(["roi", "voxelwise"]), default="roi")
@click.option("-l", "--label", type=int, default=None)
@click.option("--label-col", default=None)
@click.option("--subject-id-col", default=None)
@click.option("--modality", default=None)
@click.option("-n", "--n-jobs", type=int, default=1,
              help="Parallel workers per SLURM job")
@click.option("--strategy", type=click.Choice(["single", "array", "chunked"]),
              default="single",
              help="single: one big job. array: one SLURM task per subject. "
                   "chunked: split into N chunks.")
@click.option("--chunks", type=int, default=10,
              help="Number of chunks (only for --strategy chunked)")
@click.option("--partition", default=None, help="SLURM partition")
@click.option("--constraint", default=None,
              help="SLURM constraint (e.g. 'cpu8mem64a')")
@click.option("--time", "time_limit", default="04:00:00", help="SLURM time limit")
@click.option("--mem", default="16G", help="Memory per job")
@click.option("--conda-env", default=None,
              help="Conda environment name to activate")
@click.option("--conda-sh", default=None,
              help="Path to conda.sh (e.g. ~/miniconda3/etc/profile.d/conda.sh)")
@click.option("--script-dir", default="./slurm_scripts/",
              help="Where to write SLURM scripts")
def generate_slurm(subjects, image_col, mask_col, preset, config, output_dir,
                   mode, label, label_col, subject_id_col, modality, n_jobs,
                   strategy, chunks, partition, constraint, time_limit, mem,
                   conda_env, conda_sh, script_dir):
    """Generate SLURM submission scripts for cluster extraction."""
    from radiomicviz._slurm import generate_slurm_scripts

    paths = generate_slurm_scripts(
        subjects_csv=subjects,
        image_col=image_col,
        mask_col=mask_col,
        preset=preset,
        config=config,
        output_dir=output_dir,
        mode=mode,
        label=label,
        label_col=label_col,
        subject_id_col=subject_id_col,
        modality=modality,
        n_jobs=n_jobs,
        strategy=strategy,
        chunks=chunks,
        partition=partition,
        constraint=constraint,
        time_limit=time_limit,
        mem=mem,
        conda_env=conda_env,
        conda_sh=conda_sh,
        script_dir=script_dir,
    )
    click.echo(f"Generated {len(paths)} SLURM script(s) in {script_dir}/")
    for p in paths:
        click.echo(f"  {p}")
    click.echo(f"\nSubmit with: sbatch {paths[0]}")


if __name__ == "__main__":
    cli()
