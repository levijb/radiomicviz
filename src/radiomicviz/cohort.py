"""Generate a cohort CSV compatible with RadiomicViz batch_extract.

Usage:
    python generate_cohort_csv.py \
        --study-folder /path/to/Zenodo_MRI_dataset \
        --output-csv-name cohort

Then run extraction with:
    radiomicviz batch-extract \
        --subjects cohort.csv \
        --image-col Image \
        --mask-col Mask \
        --preset mri-default \
        --n-jobs 8 \
        --output-dir ./radiomics_output/
"""

import os
import csv
import argparse
import glob


def generate_cohort_csv(study_folder, output_csv_name):
    csv_rows = []

    subjects_path = os.path.join(study_folder, "Subjects")

    if not os.path.isdir(subjects_path):
        print(f"ERROR: Subjects directory not found at {subjects_path}")
        return

    for subject in sorted(os.listdir(subjects_path)):
        subject_path = os.path.join(subjects_path, subject)

        if not os.path.isdir(subject_path):
            continue

        for session in sorted(os.listdir(subject_path)):
            session_path = os.path.join(subject_path, session)

            if not os.path.isdir(session_path):
                continue

            # Path to segmentation folder
            seg_path = os.path.join(session_path, "derivatives", "segmentation")

            if not os.path.isdir(seg_path):
                print(f"WARNING: Segmentation directory not found for {subject}/{session}")
                continue

            # Find all mask files in segmentation folder
            mask_files = glob.glob(os.path.join(seg_path, "*.nii.gz"))

            if not mask_files:
                print(f"WARNING: No mask files found for {subject}/{session} in {seg_path}")
                continue

            # Find T1 image
            t1_pattern = os.path.join(
                session_path,
                "T1",
                f"{subject}*_T1_lesion_filled_combined_mask_bet_n4_nu.nii.gz",
            )
            t1_files = [
                f
                for f in glob.glob(t1_pattern)
                if "mask" not in os.path.basename(f).lower()
                and "seg" not in os.path.basename(f).lower()
            ]

            if not t1_files:
                print(f"WARNING: No T1 image found for {subject}/{session}")
                continue

            t1_path = t1_files[0]

            # One row per mask file
            for mask_path in sorted(mask_files):
                mask_name = os.path.splitext(
                    os.path.splitext(os.path.basename(mask_path))[0]
                )[0]  # strip .nii.gz
                print(f"Found: {subject}/{session} — {os.path.basename(mask_path)}")
                csv_rows.append([subject, session, mask_name, t1_path, mask_path])

    # Write CSV
    os.makedirs(study_folder, exist_ok=True)
    csv_path = os.path.join(study_folder, output_csv_name + ".csv")

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["subject_id", "session", "mask_name", "Image", "Mask"])
        writer.writerows(csv_rows)

    print(f"\n{'=' * 50}")
    print(f"CSV file created: {csv_path}")
    print(f"Total rows: {len(csv_rows)}")
    print(f"{'=' * 50}")
    print(f"\nRun extraction with:")
    print(f"  radiomicviz batch-extract \\")
    print(f"    --subjects {csv_path} \\")
    print(f"    --image-col Image \\")
    print(f"    --mask-col Mask \\")
    print(f"    --preset mri-default \\")
    print(f"    --n-jobs 8 \\")
    print(f"    --output-dir ./radiomics_output/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a RadiomicViz-compatible cohort CSV from a BIDS-like study folder."
    )
    parser.add_argument(
        "--study-folder",
        type=str,
        required=True,
        help="Path to the study folder (e.g., Zenodo_MRI_dataset)",
    )
    parser.add_argument(
        "--output-csv-name",
        type=str,
        required=True,
        help="Name of the output CSV file (without .csv extension)",
    )

    args = parser.parse_args()
    generate_cohort_csv(args.study_folder, args.output_csv_name)