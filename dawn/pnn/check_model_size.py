"""
Check parameter distribution of PNN models
"""
import torch
import torch.nn as nn
from pnn.models.pnn import PlasticNeuralNetwork

def count_parameters(model, name="Model"):
    """Count parameters by component"""
    print(f"\n{'='*60}")
    print(f"{name} Parameter Analysis")
    print(f"{'='*60}\n")

    # Embeddings
    emb_params = sum(p.numel() for p in model.token_embeddings.parameters())
    emb_params += sum(p.numel() for p in model.position_embeddings.parameters())
    emb_params += sum(p.numel() for p in model.embedding_layer_norm.parameters())
    print(f"Embeddings:          {emb_params:>12,} ({emb_params/1e6:>6.2f}M)")

    # DeltaRefiner(s)
    if hasattr(model, 'delta_refiner'):
        refiner = model.delta_refiner

        # Attention
        attn_params = sum(p.numel() for p in refiner.attention.parameters())
        attn_params += sum(p.numel() for p in refiner.attn_layer_norm.parameters())
        print(f"  Attention:         {attn_params:>12,} ({attn_params/1e6:>6.2f}M)")

        # FFN
        ffn_params = sum(p.numel() for p in refiner.ffn.parameters())
        ffn_params += sum(p.numel() for p in refiner.ffn_layer_norm.parameters())
        print(f"  FFN:               {ffn_params:>12,} ({ffn_params/1e6:>6.2f}M)")

        # Gate
        gate_params = sum(p.numel() for p in refiner.gate.parameters())
        print(f"  Gate:              {gate_params:>12,} ({gate_params/1e6:>6.2f}M)")

        refiner_total = attn_params + ffn_params + gate_params
        print(f"Refiner Total:       {refiner_total:>12,} ({refiner_total/1e6:>6.2f}M)")

    elif hasattr(model, 'delta_refiner1'):
        # Dual refiners
        refiner1_params = sum(p.numel() for p in model.delta_refiner1.parameters())
        refiner2_params = sum(p.numel() for p in model.delta_refiner2.parameters())
        print(f"Refiner 1:           {refiner1_params:>12,} ({refiner1_params/1e6:>6.2f}M)")
        print(f"Refiner 2:           {refiner2_params:>12,} ({refiner2_params/1e6:>6.2f}M)")
        refiner_total = refiner1_params + refiner2_params
        print(f"Refiners Total:      {refiner_total:>12,} ({refiner_total/1e6:>6.2f}M)")

    # MLM head
    mlm_params = sum(p.numel() for p in model.mlm_head.parameters())
    print(f"MLM Head:            {mlm_params:>12,} ({mlm_params/1e6:>6.2f}M)")

    # Total
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'─'*60}")
    print(f"TOTAL:               {total_params:>12,} ({total_params/1e6:>6.2f}M)")
    print(f"{'='*60}\n")

    return total_params


if __name__ == "__main__":
    print("\n" + "="*60)
    print("PNN Model Parameter Analysis")
    print("="*60)

    # Baseline PNN
    model_pnn = PlasticNeuralNetwork()
    total_pnn = count_parameters(model_pnn, "Baseline PNN")

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"PlasticNeuralNetwork: {total_pnn:>12,} ({total_pnn/1e6:>6.2f}M)")
    print(f"{'='*60}\n")
