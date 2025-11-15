"""
PNN MLM Checkpoint → DAWN Expert 변환

MLM 학습된 PNN 체크포인트를 DAWN Phase 1 expert 형태로 변환합니다.
"""

import torch
from pathlib import Path
import argparse


def convert_pnn_to_expert(
    pnn_checkpoint_path: str,
    output_path: str,
    expert_name: str = 'lexical',
    accuracy: float = None,
):
    """
    PNN MLM checkpoint를 DAWN expert checkpoint로 변환

    Args:
        pnn_checkpoint_path: PNN MLM 체크포인트 경로
        output_path: 출력 경로
        expert_name: Expert 이름
        accuracy: MLM accuracy (None이면 체크포인트에서 추출)
    """
    print("=" * 70)
    print("PNN MLM → DAWN EXPERT CONVERSION")
    print("=" * 70)
    print(f"Input:  {pnn_checkpoint_path}")
    print(f"Output: {output_path}")
    print(f"Expert: {expert_name}")
    print("-" * 70)

    # 1. PNN 체크포인트 로드
    print("\n📂 Loading PNN checkpoint...")
    ckpt = torch.load(pnn_checkpoint_path, map_location='cpu', weights_only=False)

    # 2. 메타데이터 추출
    mlm_acc = accuracy or ckpt.get('mlm_acc') or ckpt.get('accuracy') or ckpt.get('best_acc', 0.0)
    step = ckpt.get('step', 0)
    epoch = ckpt.get('epoch', 0)

    print(f"✅ Loaded PNN checkpoint")
    print(f"   MLM Accuracy: {mlm_acc:.2%}")
    print(f"   Epoch: {epoch} | Step: {step}")

    # 3. Model config 추출/생성
    model_config = ckpt.get('model_config', {})
    if not model_config:
        print("\n⚠️  No model_config found, creating default...")
        model_config = {
            'hidden_size': 768,
            'vocab_size': 30522,
            'num_heads': 12,
            'intermediate_size': [1024, 1536, 2048, 1536, 1024],  # Mountain
            'num_steps': 4,
            'num_blocks': 5,
            'dropout': 0.1,
            'max_length': 512,
        }
        print("   Using default mountain-shaped config")

    # 4. Extract model state
    model_state_raw = ckpt.get('model_state_dict', ckpt)

    # Handle position embeddings size mismatch in expert state
    target_max_length = model_config.get('max_length', 512)
    model_state = {}

    for key, tensor in model_state_raw.items():
        # Resize position embeddings if needed
        if key == 'position_embeddings.weight':
            src_len, hidden_size = tensor.shape
            if src_len != target_max_length:
                print(f"\n🔧 Resizing expert position embeddings: {src_len} → {target_max_length}")
                import math
                num_repeats = math.ceil(target_max_length / src_len)
                repeated = tensor.repeat(num_repeats, 1)
                tensor = repeated[:target_max_length]
                print(f"   ✅ Expert position embeddings resized via tiling")

        model_state[key] = tensor

    # 5. Extract shared embeddings (DAWN format)
    print(f"\n🔍 Extracting shared embeddings...")
    shared_embeddings_state = {}
    embedding_keys = {
        'token_embeddings.weight': 'token.weight',
        'position_embeddings.weight': 'position.weight',
        'embedding_layer_norm.weight': 'layer_norm.weight',
        'embedding_layer_norm.bias': 'layer_norm.bias',
    }

    target_max_length = model_config.get('max_length', 512)

    for pnn_key, dawn_key in embedding_keys.items():
        if pnn_key in model_state:
            tensor = model_state[pnn_key]

            # Handle position embeddings size mismatch (e.g., PNN 128 → DAWN 512)
            if pnn_key == 'position_embeddings.weight':
                src_len, hidden_size = tensor.shape
                if src_len != target_max_length:
                    print(f"   ⚠️  Position embeddings size mismatch: {src_len} → {target_max_length}")

                    # Safe tiling strategy: repeat learned positions
                    import math
                    num_repeats = math.ceil(target_max_length / src_len)
                    repeated = tensor.repeat(num_repeats, 1)  # [num_repeats * src_len, hidden_size]
                    tensor = repeated[:target_max_length]  # Trim to exact target length

                    print(f"   🔧 Tiled position embeddings {src_len} → {target_max_length} (cyclic repeat)")
                    print(f"   ✅ All positions now have learned values (no random init)")

            shared_embeddings_state[dawn_key] = tensor
            print(f"   ✓ {pnn_key} -> {dawn_key} {list(tensor.shape)}")
        else:
            print(f"   ⚠️  {pnn_key} not found in checkpoint")

    print(f"✅ Extracted {len(shared_embeddings_state)} embedding tensors")

    # 6. DAWN expert checkpoint 생성 (새로운 config 구조)
    expert_ckpt = {
        # Phase 1 metadata
        'phase': 1,
        'expert_name': expert_name,
        'task': 'mlm',  # MLM으로 pre-trained
        'epoch': epoch,
        'step': step,
        'is_final': True,  # MLM 학습 완료

        # Model state (DAWN format)
        'expert_state_dict': model_state,
        'shared_embeddings_state_dict': shared_embeddings_state,
        'task_heads_state_dict': {},  # Will be re-initialized

        # Metrics
        'accuracy': mlm_acc,
        'mlm_acc': mlm_acc,
        'best_acc': mlm_acc,
        'metrics': {
            'epoch': epoch,
            'train_loss': ckpt.get('loss', 0.0),
            'train_accuracy': mlm_acc * 100,  # Convert to percentage
            'val_loss': ckpt.get('loss', 0.0),
            'val_accuracy': mlm_acc * 100,
            'time': 0.0,
        },
        'training_history': [],

        # Model config (expert_loader 호환)
        'model_config': model_config,

        # Training state (don't transfer - will be re-initialized)
        'optimizer_state_dict': None,
        'scheduler_state_dict': None,

        # History
        'history': ckpt.get('history', {}),
    }

    # 7. 저장
    print(f"\n💾 Saving DAWN expert checkpoint...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(expert_ckpt, output_path)

    file_size = output_path.stat().st_size / (1024 * 1024)  # MB
    print(f"✅ Saved: {output_path.name} ({file_size:.1f} MB)")

    # 8. 검증
    print(f"\n🔍 Validating converted checkpoint...")
    test_ckpt = torch.load(output_path, map_location='cpu')

    required_keys = ['phase', 'expert_name', 'expert_state_dict', 'shared_embeddings_state_dict', 'accuracy', 'model_config']
    missing = [k for k in required_keys if k not in test_ckpt]

    if missing:
        print(f"❌ Missing keys: {missing}")
    else:
        print(f"✅ All required keys present")
        print(f"   • expert_state_dict: {len(test_ckpt['expert_state_dict'])} parameters")
        print(f"   • shared_embeddings_state_dict: {len(test_ckpt['shared_embeddings_state_dict'])} tensors")

    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE!")
    print("=" * 70)
    print(f"Expert: {expert_name}")
    print(f"MLM Accuracy: {mlm_acc:.2%}")
    print(f"Output: {output_path}")
    print("\n💡 Next step: Load with expert_loader and start curriculum training")
    print("=" * 70)

    return expert_ckpt


def main():
    parser = argparse.ArgumentParser(description='Convert PNN MLM checkpoint to DAWN expert')
    parser.add_argument('--input', type=str, required=True,
                        help='PNN MLM checkpoint path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output expert checkpoint path')
    parser.add_argument('--expert_name', type=str, default='lexical',
                        help='Expert name (default: lexical)')
    parser.add_argument('--accuracy', type=float, default=None,
                        help='MLM accuracy (if not in checkpoint)')

    args = parser.parse_args()

    convert_pnn_to_expert(
        pnn_checkpoint_path=args.input,
        output_path=args.output,
        expert_name=args.expert_name,
        accuracy=args.accuracy
    )


if __name__ == "__main__":
    """
    Usage examples:

    # Basic conversion (DAWN format)
    python scripts/convert_pnn_to_expert.py \
        --input /content/drive/MyDrive/pnn/hierarchical/best_model.pt \
        --output /content/drive/MyDrive/dawn/phase1_checkpoints/lexical_mlm_final.pt \
        --expert_name lexical

    # Convert for different expert
    python scripts/convert_pnn_to_expert.py \
        --input /content/drive/MyDrive/pnn/hierarchical/best_model.pt \
        --output /content/drive/MyDrive/dawn/phase1_checkpoints/semantic_mlm_final.pt \
        --expert_name semantic \
        --accuracy 0.85

    # Output filename format:
    # {expert_name}_{task}_final.pt
    # Example: lexical_mlm_final.pt, semantic_nli_final.pt

    Note: The converted checkpoint includes:
    - expert_state_dict: Full expert model parameters
    - shared_embeddings_state_dict: Extracted embeddings (token, position, layer_norm)
    - Proper DAWN metadata for checkpoint loading
    - No preset information (direct config customization)
    """
    main()
