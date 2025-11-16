"""
Test sparsity of Brain-Like model
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from src.models.sprout_brain_like import create_brain_like_sprout


def test_sparsity():
    """Test if sparse activation actually works"""
    print("="*70)
    print("TESTING SPARSITY")
    print("="*70)

    # Create model
    model = create_brain_like_sprout(
        vocab_size=30522,
        n_neurons=4096,
        d_state=256,
        initial_sparsity=128,
        final_sparsity=256
    )
    model.eval()

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Test different inputs
    test_texts = [
        "The cat is sleeping on the mat",
        "I love programming in Python",
        "Quantum mechanics is very difficult"
    ]

    print("\n1. Testing initial activation patterns:")
    print("-" * 70)

    with torch.no_grad():
        for i, text in enumerate(test_texts):
            tokens = tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                max_length=32,
                truncation=True
            )['input_ids']

            # Get initial activation
            activation = model.input_encoder(tokens)[0]  # [n_neurons]

            # Statistics
            n_total = activation.shape[0]
            n_nonzero = (activation > 0.0).sum().item()
            n_significant = (activation > 0.01).sum().item()
            n_active = (activation > 0.1).sum().item()
            n_strong = (activation > 0.5).sum().item()

            max_val = activation.max().item()
            min_val = activation.min().item()
            mean_val = activation.mean().item()

            print(f"\nInput {i+1}: \"{text}\"")
            print(f"  Non-zero (>0.0):      {n_nonzero:4d} / {n_total}")
            print(f"  Significant (>0.01):  {n_significant:4d} / {n_total}")
            print(f"  Active (>0.1):        {n_active:4d} / {n_total}")
            print(f"  Strong (>0.5):        {n_strong:4d} / {n_total}")
            print(f"  Max: {max_val:.4f}, Min: {min_val:.4f}, Mean: {mean_val:.4f}")

            # Top active neurons
            top_values, top_indices = torch.topk(activation, k=10)
            print(f"  Top 10 neurons: {top_indices.tolist()}")
            print(f"  Top 10 values:  {[f'{v:.3f}' for v in top_values.tolist()]}")

    print("\n2. Testing diversity:")
    print("-" * 70)

    activations = []
    with torch.no_grad():
        for text in test_texts:
            tokens = tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                max_length=32,
                truncation=True
            )['input_ids']

            act = model.input_encoder(tokens)[0]
            activations.append(act)

    # Pairwise similarity
    for i in range(len(activations)):
        for j in range(i+1, len(activations)):
            sim = F.cosine_similarity(
                activations[i].unsqueeze(0),
                activations[j].unsqueeze(0)
            ).item()
            print(f"  Similarity {i+1} vs {j+1}: {sim:.4f}")

    print("\n3. Expected vs Actual:")
    print("-" * 70)
    print(f"  Expected sparsity: 128-256 active neurons")
    print(f"  Actual (>0.1):     {n_active} neurons")
    print(f"  Expected max:      ~1.0")
    print(f"  Actual max:        {max_val:.4f}")
    print(f"  Expected diversity: <0.7 similarity")
    print(f"  Actual diversity:   {sim:.4f} similarity")

    if n_active > 512:
        print("\n❌ SPARSITY FAILED: Too many active neurons!")
    elif n_active < 64:
        print("\n❌ SPARSITY FAILED: Too few active neurons!")
    else:
        print("\n✅ Sparsity OK")

    if sim > 0.8:
        print("❌ DIVERSITY FAILED: Patterns too similar!")
    else:
        print("✅ Diversity OK")

    print("\n" + "="*70)


if __name__ == "__main__":
    test_sparsity()
