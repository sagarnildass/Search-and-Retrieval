"""
Standalone script to zip and prepare checkpoints for download from Kaggle

Add this as a separate cell in your Kaggle notebook after training completes.
"""

import shutil
from pathlib import Path

# Configuration
CHECKPOINT_DIR = "/kaggle/working/checkpoints"
ZIP_NAME = "checkpoints_stage1.zip"

print("=" * 80)
print("CREATING CHECKPOINT ARCHIVE FOR DOWNLOAD")
print("=" * 80)

checkpoint_dir = Path(CHECKPOINT_DIR)
zip_path = Path("/kaggle/working") / ZIP_NAME

if not checkpoint_dir.exists():
    print(f"\nError: Checkpoint directory not found: {CHECKPOINT_DIR}")
    print(f"   Make sure training has completed and checkpoints were saved.")
else:
    # List all checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("*.pt"))
    
    if not checkpoint_files:
        print(f"\nNo checkpoint files found in {CHECKPOINT_DIR}")
    else:
        print(f"\nFound {len(checkpoint_files)} checkpoint file(s):")
        total_size = 0
        for cp_file in sorted(checkpoint_files):
            size_mb = cp_file.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"  - {cp_file.name} ({size_mb:.2f} MB)")
        
        print(f"\nTotal size: {total_size:.2f} MB")
        
        # Create zip file
        print(f"\nCreating zip archive...")
        try:
            # Remove existing zip if it exists
            if zip_path.exists():
                zip_path.unlink()
            
            # Create zip
            shutil.make_archive(
                str(zip_path).replace('.zip', ''),  # Remove .zip (make_archive adds it)
                'zip',
                checkpoint_dir
            )
            
            zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
            print(f"\n✓ Success! Archive created: {zip_path}")
            print(f"  Archive size: {zip_size_mb:.2f} MB")
            print(f"  Compression ratio: {(1 - zip_size_mb/total_size)*100:.1f}%")
            
            print(f"\n{'='*80}")
            print("DOWNLOAD INSTRUCTIONS")
            print(f"{'='*80}")
            print(f"\n1. Go to the 'Output' tab in your Kaggle notebook")
            print(f"2. Find '{ZIP_NAME}' in the file list")
            print(f"3. Click the download button (⬇️) next to the file")
            print(f"\nAlternatively:")
            print(f"1. Click 'Save Version' → 'Save & Run All'")
            print(f"2. After completion, go to 'Output' tab")
            print(f"3. Download the zip file from there")
            
        except Exception as e:
            print(f"\nError creating zip file: {e}")
            print(f"\nManual download option:")
            print(f"1. Click 'Save Version' → 'Save & Run All'")
            print(f"2. After completion, go to 'Output' tab")
            print(f"3. The checkpoints folder will be available there")
