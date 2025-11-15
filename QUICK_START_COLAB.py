"""
SPROUT Quick Start for Google Colab

Copy and paste this entire file into a Colab cell to get started!
"""

# ============================================================================
# STEP 1: Mount Google Drive (for saving checkpoints)
# ============================================================================
from google.colab import drive
drive.mount('/content/drive')

# ============================================================================
# STEP 2: Clone Repository
# ============================================================================
# Option A: Using GitHub token (if repo is private)
# from google.colab import userdata
# token = userdata.get('GITHUB_TOKEN')
# !git clone https://x-access-token:{token}@github.com/madst0614/sprout.git

# Option B: Public clone (use this for now)
!git clone https://github.com/madst0614/sprout.git 2>/dev/null || echo "Repository already cloned"

# ============================================================================
# STEP 3: Navigate and Update
# ============================================================================
%cd /content/sprout

# Checkout the working branch
!git checkout claude/implement-sprout-model-01L9icytevoJcfrrmKZo6Rm4
!git pull origin claude/implement-sprout-model-01L9icytevoJcfrrmKZo6Rm4

# ============================================================================
# STEP 4: Install Dependencies
# ============================================================================
print("\n" + "="*70)
print("Installing Dependencies...")
print("="*70 + "\n")

!pip install -q torch transformers datasets tqdm

print("\n✅ Dependencies installed!\n")

# ============================================================================
# STEP 5: Quick Test (Debug Mode)
# ============================================================================
print("="*70)
print("QUICK TEST - Debug Mode")
print("="*70 + "\n")

!python scripts/train_sprout_mlm.py \
  --debug_mode \
  --num_epochs 1 \
  --batch_size 8 \
  --max_nodes 5 \
  --visualize_structure

# ============================================================================
# STEP 6: Full Training (Uncomment to run)
# ============================================================================
print("\n" + "="*70)
print("To run FULL TRAINING, uncomment and run the cell below")
print("="*70)
print("""
# Full Training Command:
!python scripts/train_sprout_mlm.py \\
  --checkpoint_dir /content/drive/MyDrive/sprout/checkpoints/ \\
  --num_epochs 3 \\
  --batch_size 32 \\
  --max_samples 50000 \\
  --max_nodes 5 \\
  --hidden_dim 512 \\
  --learning_rate 5e-5 \\
  --mixed_precision \\
  --visualize_structure
""")
