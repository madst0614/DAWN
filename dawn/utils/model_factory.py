"""
Model Factory - Create models from configurations

Uses new unified config system (configs module) to instantiate models.
"""

import torch
from pathlib import Path
from typing import Dict, Optional, List

from configs import (
    get_config,
    get_phase1_config,
    get_phase2_config,
    RuntimeConfig,
)


class ModelFactory:
    """Factory for creating DAWN models from configurations"""

    @staticmethod
    def create_delta_expert(
        name: str, config: Dict, peer_names: Optional[List[str]] = None
    ):
        """
        Create a single DeltaExpert from config

        Args:
            name: Expert name
            config: Model configuration dict (from RuntimeConfig.to_dict())
            peer_names: Peer expert names (Phase 2)
        Returns:
            DeltaExpert instance
        """
        from models.delta_expert import DeltaExpert

        # DeltaExpert expects full config dict
        expert = DeltaExpert(config=config, peer_names=peer_names)

        return expert

    @staticmethod
    def create_dawn(cfg: RuntimeConfig, phase: int = 1):
        """
        Create DAWN model from RuntimeConfig

        Args:
            cfg: RuntimeConfig instance
            phase: 1 or 2
        Returns:
            DAWN instance
        """
        from models import DAWN

        enable_peer = (phase == 2) or (cfg.peer_prediction is not None)

        # Convert to dict for DAWN
        config_dict = cfg.to_dict()

        # Determine integrator type based on phase
        integrator_type = "auto"  # Auto: PassThrough for Phase 1, Attention for Phase 2

        model = DAWN(
            config=config_dict,
            enable_peer_prediction=enable_peer,
            active_experts=None,  # Create all experts
            integrator_type=integrator_type,
        )

        # Print model info
        ModelFactory._print_model_info(model, cfg, phase)

        return model

    @staticmethod
    def _print_model_info(model, cfg, phase: int):
        """Print model information (supports both RuntimeConfig and DAWNConfig)"""
        # Get model structure info
        info = model.get_model_info()

        # Calculate parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n{'='*70}")
        print(f"DAWN Phase {phase} Model Created")
        print(f"{'='*70}")
        print(f"Shared Embeddings: {'✅' if info['shared_embeddings'] else '❌'} (vocab={info['vocab_size']}, dim={info['hidden_size']})")
        print(f"Active Experts: {info['active_experts']} ({len(info['active_experts'])})")
        if len(info['active_experts']) < len(info['all_experts']):
            inactive = set(info['all_experts']) - set(info['active_experts'])
            print(f"Inactive Experts: {list(inactive)}")
        print(f"Integrator: {'✅' if info.get('has_expert_integrator', False) else '❌ (Phase 1 mode)'}")
        print(f"Peer Prediction: {'✅' if info['has_peer_prediction'] else '❌'}")
        print(f"Task Heads: External (TaskHeads module)")

        print(f"\nParameters:")
        print(f"  Total: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"  Trainable: {trainable_params:,} ({trainable_params/1e6:.1f}M)")

        # Per-expert params (if experts exist)
        if info['active_experts']:
            expert_name = info['active_experts'][0]
            expert_params = sum(p.numel() for p in model.experts[expert_name].parameters())
            print(f"  Per-expert: {expert_params:,} ({expert_params/1e6:.1f}M)")

        # Phase 2 specific (check for peer_prediction attribute)
        if phase == 2 and hasattr(cfg, 'peer_prediction') and cfg.peer_prediction:
            print(f"\nPeer Prediction Config:")
            print(f"  Iterations: {cfg.peer_prediction.num_iterations}")
            print(f"  Loss weight: {cfg.peer_prediction.prediction_loss_weight}")

        print(f"{'='*70}\n")

    @staticmethod
    def load_phase1_checkpoint(expert_name: str, checkpoint_path: str, cfg: RuntimeConfig):
        """
        Load a Phase 1 expert checkpoint

        Args:
            expert_name: Expert name
            checkpoint_path: Path to checkpoint
            cfg: RuntimeConfig instance
        Returns:
            Loaded DeltaExpert
        """
        from models.delta_expert import DeltaExpert

        # Create expert
        config_dict = cfg.to_dict()
        expert = ModelFactory.create_delta_expert(expert_name, config_dict, peer_names=None)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        expert.load_state_dict(checkpoint["model_state_dict"], strict=False)

        print(f"Loaded expert '{expert_name}' from {checkpoint_path}")
        if "epoch" in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if "best_loss" in checkpoint:
            print(f"  Best loss: {checkpoint['best_loss']:.4f}")

        return expert

    @staticmethod
    def create_from_config(cfg: RuntimeConfig, device: str = "cuda"):
        """
        Create model from RuntimeConfig

        Args:
            cfg: RuntimeConfig instance
            device: Device to place model on
        Returns:
            DAWN model and config
        """
        # Print config summary
        cfg.print_summary()

        # Determine phase
        phase = cfg.phase

        # Create model
        model = ModelFactory.create_dawn(cfg, phase=phase)

        # Move to device
        model = model.to(device)

        return model, cfg

    @staticmethod
    def create_phase1_model(
        model_preset: str = "base",
        expert_preset: str = "standard",
        training_preset: str = "standard",
        expert_names: Optional[List[str]] = None,
        device: str = "cuda",
    ):
        """
        Create Phase 1 model with specified config

        Args:
            model_preset: Model size preset (tiny, small, base, large)
            expert_preset: Expert config preset (standard, efficient, deep_refinement)
            training_preset: Training preset (standard, fast, conservative)
            expert_names: List of expert names (None = all)
            device: Device to place model on
        Returns:
            DAWN model and RuntimeConfig
        """
        cfg = get_phase1_config(
            model_preset=model_preset,
            expert_preset=expert_preset,
            training_preset=training_preset,
            expert_names=expert_names,
        )

        cfg.print_summary()

        model = ModelFactory.create_dawn(cfg, phase=1)
        model = model.to(device)

        return model, cfg

    @staticmethod
    def create_phase2_model(
        checkpoint_paths: Dict[str, str],
        model_preset: str = "base",
        peer_prediction_preset: str = "standard",
        training_preset: str = "standard",
        device: str = "cuda",
    ):
        """
        Create Phase 2 model from Phase 1 checkpoints

        Args:
            checkpoint_paths: {expert_name: checkpoint_path}
            model_preset: Model size preset (tiny, small, base, large)
            peer_prediction_preset: Peer prediction config preset (standard, aggressive, conservative)
            training_preset: Training preset (standard, aggressive, conservative)
            device: Device to place model on
        Returns:
            DAWN model and RuntimeConfig
        """
        from models import DAWN

        expert_names = list(checkpoint_paths.keys())

        cfg = get_phase2_config(
            model_preset=model_preset,
            peer_prediction_preset=peer_prediction_preset,
            training_preset=training_preset,
            expert_names=expert_names,
        )

        cfg.print_summary()

        # Convert to dict for DAWN
        config_dict = cfg.to_dict()

        # Create model with peer prediction enabled
        model = DAWN(
            config=config_dict,
            enable_peer_prediction=True,
            active_experts=None,  # Create all experts
            integrator_type="auto",  # Will use Attention integrator for Phase 2
        )

        # Load expert checkpoints
        for name, path in checkpoint_paths.items():
            checkpoint = torch.load(path, map_location="cpu")

            # Load weights (excluding peer predictors)
            expert_state = {
                k: v
                for k, v in checkpoint["model_state_dict"].items()
                if not k.startswith("peer_predictors")
            }

            model.experts[name].load_state_dict(expert_state, strict=False)
            print(f"Loaded expert '{name}' from {path}")

        ModelFactory._print_model_info(model, cfg, phase=2)

        # Move to device
        model = model.to(device)

        return model, cfg


def create_model_from_config(cfg: RuntimeConfig, device: str = "cuda"):
    """
    Convenience function to create model from RuntimeConfig

    Args:
        cfg: RuntimeConfig instance
        device: Device
    Returns:
        model, config
    """
    return ModelFactory.create_from_config(cfg, device)


def create_phase1_model(model_preset: str = "base", device: str = "cuda"):
    """
    Convenience function for Phase 1 model

    Args:
        model_preset: Model size preset (tiny, small, base, large)
        device: Device
    Returns:
        model, RuntimeConfig
    """
    return ModelFactory.create_phase1_model(model_preset=model_preset, device=device)


# ============================================================================
# TESTING
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Testing Model Factory")
    print("=" * 70)

    # Test 1: Phase 1 with base preset
    print("\n[Test 1] Creating Phase 1 base model...")
    cfg_p1 = get_phase1_config(model_preset="base")
    model_p1, _ = ModelFactory.create_from_config(cfg_p1, device="cpu")

    # Test forward pass
    print("\nTesting forward pass...")
    batch = torch.randint(0, 30522, (2, 128))
    mask = torch.ones(2, 128)
    output = model_p1(batch, mask)
    print(f"✓ Output shape: {output.shape}")

    # Test 2: Phase 1 with tiny config
    print("\n[Test 2] Creating tiny model for development...")
    model_tiny, config_tiny = create_phase1_model("tiny", device="cpu")
    print(f"✓ Tiny model created: {config_tiny.model.hidden_size}d")

    # Test 3: Show available presets
    print("\n[Test 3] Available presets:")
    from configs import MODEL_PRESETS, EXPERT_PRESETS

    print("  Model sizes:", ", ".join(MODEL_PRESETS.keys()))
    print("  Expert configs:", ", ".join(EXPERT_PRESETS.keys()))
    print("  Training presets: Removed - use get_default_config() with direct parameters")

    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
