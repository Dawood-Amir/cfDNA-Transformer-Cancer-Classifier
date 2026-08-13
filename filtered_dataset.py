import os
import shutil
import pandas as pd
import torch
from tqdm import tqdm
import argparse


def filter_dataset(
    input_dir: str,
    output_dir: str,
    classes_to_keep: list = [0, 1, 2],  # Healthy=0, GBM=1, LGG=2 (keep), DMG=3 (drop)
    copy_tensors: bool = True
):
    """
    Filter a PatientDataset by keeping only specified classes.
    
    Args:
        input_dir: Path to the original patient_tensors folder
        output_dir: Path where filtered data will be saved
        classes_to_keep: List of class labels to keep (default: [0, 1, 2] = Healthy, GBM, LGG)
        copy_tensors: If True, copy tensor files; if False, create symlinks (saves space)
    """
    
    print("=" * 70)
    print("📂 FILTERING DATASET (Removing Class 3 - DMG)")
    print("=" * 70)
    
    # --- Load manifest ---
    manifest_path = os.path.join(input_dir, "manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at: {manifest_path}")
    
    manifest = pd.read_csv(manifest_path)
    print(f"\n📊 Original dataset:")
    print(f"  Total patients: {len(manifest)}")
    print(f"  Class distribution:")
    for label, count in manifest['label'].value_counts().sort_index().items():
        class_name = {0: 'Healthy', 1: 'GBM', 2: 'LGG', 3: 'DMG_H3K27M'}.get(label, f'Class_{label}')
        print(f"    {class_name}: {count}")
    
    # --- Filter rows ---
    filtered_manifest = manifest[manifest['label'].isin(classes_to_keep)]
    
    print(f"\n🔍 Filtering to keep classes: {classes_to_keep}")
    print(f"  Filtered patients: {len(filtered_manifest)}")
    print(f"  Removed patients: {len(manifest) - len(filtered_manifest)}")
    
    print(f"\n  New class distribution:")
    for label, count in filtered_manifest['label'].value_counts().sort_index().items():
        class_name = {0: 'Healthy', 1: 'GBM', 2: 'LGG', 3: 'DMG_H3K27M'}.get(label, f'Class_{label}')
        print(f"    {class_name}: {count}")
    
    # --- Create output directory ---
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Copy or symlink tensor files ---
    print(f"\n📁 Copying tensor files to: {output_dir}")
    
    copied_count = 0
    failed_count = 0
    
    # Use tqdm for progress bar
    for idx, row in tqdm(filtered_manifest.iterrows(), total=len(filtered_manifest), desc="Copying files"):
        filename = row['filename']
        src_path = os.path.join(input_dir, filename)
        dst_path = os.path.join(output_dir, filename)
        
        try:
            if copy_tensors:
                # Copy the file (actual copy)
                shutil.copy2(src_path, dst_path)
            else:
                # Create a symbolic link (saves disk space, but requires read access to original)
                if os.path.exists(dst_path):
                    os.remove(dst_path)
                os.symlink(src_path, dst_path)
            copied_count += 1
        except Exception as e:
            print(f"  ⚠️ Failed to copy {filename}: {e}")
            failed_count += 1
    
    # --- Save filtered manifest ---
    filtered_manifest_path = os.path.join(output_dir, "manifest.csv")
    filtered_manifest.to_csv(filtered_manifest_path, index=False)
    
    print(f"\n✅ Done!")
    print(f"  Copied: {copied_count} files")
    print(f"  Failed: {failed_count} files")
    print(f"  Manifest saved: {filtered_manifest_path}")
    print(f"\n📁 Filtered dataset ready at: {output_dir}")
    
    # --- Verify ---
    print(f"\n🔍 Verifying filtered dataset...")
    test_dataset = pd.read_csv(filtered_manifest_path)
    print(f"  Total patients in filtered manifest: {len(test_dataset)}")
    print(f"  Classes present: {sorted(test_dataset['label'].unique())}")
    print(f"  Class distribution:")
    for label, count in test_dataset['label'].value_counts().sort_index().items():
        class_name = {0: 'Healthy', 1: 'GBM', 2: 'LGG', 3: 'DMG_H3K27M'}.get(label, f'Class_{label}')
        print(f"    {class_name}: {count}")
    
    print("\n" + "=" * 70)
    print("🎉 FILTERING COMPLETE!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Filter dataset to keep only specified classes")
    parser.add_argument('--input_dir', type=str, 
                        default='src/data/processed/patient_tensors',
                        help='Path to original patient_tensors folder')
    parser.add_argument('--output_dir', type=str,
                        default='src/data/processed/patient_tensors_3classes',
                        help='Path where filtered data will be saved')
    parser.add_argument('--classes', type=int, nargs='+', default=[0, 1, 2],
                        help='Class labels to keep (default: 0 1 2 = Healthy, GBM, LGG)')
    parser.add_argument('--symlink', action='store_true',
                        help='Use symlinks instead of copying (saves disk space)')
    
    args = parser.parse_args()
    
    # Resolve paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, args.input_dir)
    output_dir = os.path.join(script_dir, args.output_dir)
    
    print(f"Script directory: {script_dir}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    if not os.path.exists(input_dir):
        print(f"❌ Error: Input directory not found: {input_dir}")
        return
    
    filter_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        classes_to_keep=args.classes,
        copy_tensors=not args.symlink
    )


if __name__ == "__main__":
    main()