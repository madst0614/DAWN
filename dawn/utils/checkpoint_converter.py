"""
Checkpoint Converter: models.* → models.*

Converts checkpoints from old models/ structure to models/ structure.
In most cases, checkpoints are compatible without conversion since the
architecture is identical. This tool validates and handles edge cases.
"""

import torch
from typing import Dict, Any
from pathlib import Path


def validate_checkpoint_compatibility(checkpoint_path: str) -> bool:
    """
    Validate if checkpoint is compatible with models package

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        True if compatible, False otherwise
    """
    print(f"\n{'='*70}")
    print("CHECKPOINT COMPATIBILITY CHECK")
    print(f"{'='*70}\n")

    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return False

    # Check required keys
    required_keys = ['model_state_dict', 'phase']
    missing = [k for k in required_keys if k not in ckpt]
    if missing:
        print(f"❌ Missing required keys: {missing}")
        return False

    # Check model state dict structure
    model_state = ckpt['model_state_dict']

    # Expected patterns for models structure
    expected_patterns = [
        'experts.',           # Expert modules
        'shared_embeddings.', # Shared embeddings
    ]

    # Check if keys match expected patterns
    has_expected = False
    for key in model_state.keys():
        if any(pattern in key for pattern in expected_patterns):
            has_expected = True
            break

    if not has_expected:
        print(f"⚠️  Warning: Unexpected state dict structure")
        print(f"    First few keys: {list(model_state.keys())[:5]}")
        return False

    # Validation passed
    print(f"✅ Checkpoint structure looks compatible!")
    print(f"   Phase: {ckpt['phase']}")
    print(f"   Experts: {ckpt.get('expert_names', 'N/A')}")
    print(f"   Model keys: {len(model_state)} parameters")

    if 'task_heads_state_dict' in ckpt:
        print(f"   Task heads: ✓ Found")
    else:
        print(f"   Task heads: ✗ Not found")

    print(f"\n{'='*70}\n")
    return True


def convert_checkpoint(
    old_checkpoint_path: str,
    new_checkpoint_path: str,
    dry_run: bool = False,
) -> bool:
    """
    Convert checkpoint from old to new format

    In practice, most checkpoints are already compatible because the
    architecture hasn't changed, only the code organization.

    Args:
        old_checkpoint_path: Path to old checkpoint
        new_checkpoint_path: Path to save converted checkpoint
        dry_run: If True, don't save, just validate

    Returns:
        True if conversion successful
    """
    print(f"\n{'='*70}")
    print("CHECKPOINT CONVERSION")
    print(f"{'='*70}\n")

    print(f"📂 Loading: {old_checkpoint_path}")

    try:
        ckpt = torch.load(old_checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return False

    # In most cases, no conversion needed!
    # The state_dict keys should be identical between models/ and dawn_v4/

    # Validate structure
    if not validate_checkpoint_compatibility(old_checkpoint_path):
        print("❌ Checkpoint validation failed")
        return False

    if dry_run:
        print("✅ Dry run successful - checkpoint is compatible")
        print("   No conversion needed!")
        return True

    # Save (even if no changes, this creates a backup)
    print(f"💾 Saving: {new_checkpoint_path}")
    torch.save(ckpt, new_checkpoint_path)
    print(f"✅ Checkpoint saved successfully")

    print(f"\n{'='*70}\n")
    return True


def batch_convert_checkpoints(
    checkpoint_dir: str,
    output_dir: str,
    pattern: str = "*.pt",
    dry_run: bool = False,
):
    """
    Convert all checkpoints in a directory

    Args:
        checkpoint_dir: Directory containing old checkpoints
        output_dir: Directory to save converted checkpoints
        pattern: File pattern to match (default: "*.pt")
        dry_run: If True, validate only, don't save
    """
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)

    if not checkpoint_dir.exists():
        print(f"❌ Directory not found: {checkpoint_dir}")
        return

    # Find all checkpoint files
    checkpoint_files = list(checkpoint_dir.glob(pattern))

    if not checkpoint_files:
        print(f"❌ No checkpoint files found matching '{pattern}'")
        return

    print(f"\n{'='*70}")
    print(f"BATCH CHECKPOINT CONVERSION")
    print(f"{'='*70}\n")
    print(f"Input dir:  {checkpoint_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Pattern:    {pattern}")
    print(f"Files:      {len(checkpoint_files)}")
    print(f"Dry run:    {dry_run}")
    print(f"{'='*70}\n")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Process each file
    success_count = 0
    for ckpt_file in checkpoint_files:
        print(f"\n📄 Processing: {ckpt_file.name}")

        new_path = output_dir / ckpt_file.name

        if convert_checkpoint(str(ckpt_file), str(new_path), dry_run=dry_run):
            success_count += 1
        else:
            print(f"❌ Failed: {ckpt_file.name}")

    # Summary
    print(f"\n{'='*70}")
    print(f"CONVERSION SUMMARY")
    print(f"{'='*70}\n")
    print(f"Total:      {len(checkpoint_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed:     {len(checkpoint_files) - success_count}")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert DAWN checkpoints")
    parser.add_argument("checkpoint_path", help="Path to checkpoint file or directory")
    parser.add_argument("--output", "-o", help="Output path (default: same as input with _v4 suffix)")
    parser.add_argument("--batch", action="store_true", help="Convert all checkpoints in directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, don't save")
    parser.add_argument("--pattern", default="*.pt", help="File pattern for batch mode")

    args = parser.parse_args()

    input_path = Path(args.checkpoint_path)

    if args.batch:
        # Batch mode
        output_dir = args.output or str(input_path.parent / f"{input_path.name}_v4")
        batch_convert_checkpoints(
            checkpoint_dir=str(input_path),
            output_dir=output_dir,
            pattern=args.pattern,
            dry_run=args.dry_run,
        )
    else:
        # Single file mode
        if not input_path.exists():
            print(f"❌ File not found: {input_path}")
            exit(1)

        output_path = args.output or str(input_path.parent / f"{input_path.stem}_v4{input_path.suffix}")

        success = convert_checkpoint(
            old_checkpoint_path=str(input_path),
            new_checkpoint_path=output_path,
            dry_run=args.dry_run,
        )

        if success:
            print("✅ Done!")
        else:
            print("❌ Conversion failed")
            exit(1)
