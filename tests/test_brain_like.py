"""
Tests for Brain-Like SPROUT Architecture

뇌 기반 아키텍처의 핵심 기능 테스트
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from src.models.sprout_brain_like import (
    GlobalNeuronPool,
    InputToActivation,
    NeuronState,
    NeuronInteraction,
    OutputDecoder,
    SPROUT_BrainLike,
    create_brain_like_sprout
)


def test_global_neuron_pool():
    """Test GlobalNeuronPool creation and basic operations."""
    n_neurons = 512
    d_state = 128

    pool = GlobalNeuronPool(n_neurons=n_neurons, d_state=d_state)

    # Check shapes
    assert pool.neuron_signatures.shape == (n_neurons, d_state)
    assert pool.connection_strength.shape == (n_neurons, n_neurons)

    # Test get_signatures
    indices = torch.tensor([0, 10, 20, 30])
    signatures = pool.get_signatures(indices)
    assert signatures.shape == (len(indices), d_state)

    # Test get_connections
    connections = pool.get_connections(indices)
    assert connections.shape == (len(indices), len(indices))

    print("✅ GlobalNeuronPool test passed")


def test_neuron_state():
    """Test NeuronState creation."""
    n_neurons = 512
    d_state = 128

    state = NeuronState.create(n_neurons, d_state, torch.device('cpu'))

    assert state.activation.shape == (n_neurons,)
    assert state.hidden_state.shape == (n_neurons, d_state)
    assert state.activation.sum().item() == 0  # Initially all zeros
    assert state.hidden_state.sum().item() == 0

    print("✅ NeuronState test passed")


def test_input_to_activation():
    """Test InputToActivation module."""
    vocab_size = 1000
    n_neurons = 512
    d_model = 128
    batch_size = 2
    seq_len = 10

    encoder = InputToActivation(
        vocab_size=vocab_size,
        d_model=d_model,
        n_neurons=n_neurons,
        top_k=64
    )

    # Test forward
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    activation = encoder(tokens)

    # Check shape
    assert activation.shape == (batch_size, n_neurons)

    # Check sparsity (should have exactly top_k non-zero)
    for i in range(batch_size):
        n_active = (activation[i] > 0).sum().item()
        assert n_active == 64, f"Expected 64 active neurons, got {n_active}"

    # Check values in [0, 1]
    assert (activation >= 0).all()
    assert (activation <= 1).all()

    print("✅ InputToActivation test passed")


def test_neuron_interaction():
    """Test NeuronInteraction module."""
    n_neurons = 512
    d_state = 128

    interaction = NeuronInteraction(n_neurons=n_neurons, d_state=d_state)

    # Create initial state with some active neurons
    state = NeuronState.create(n_neurons, d_state, torch.device('cpu'))

    # Activate some neurons
    active_indices = torch.tensor([0, 10, 20, 30, 40])
    state.activation[active_indices] = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
    state.hidden_state[active_indices] = torch.randn(len(active_indices), d_state)

    # Test interaction
    new_state = interaction(state, sparsity_k=100)

    # Check that we got a new state
    assert isinstance(new_state, NeuronState)
    assert new_state.activation.shape == (n_neurons,)
    assert new_state.hidden_state.shape == (n_neurons, d_state)

    # Check sparsity maintained
    n_active = (new_state.activation > 0.01).sum().item()
    assert n_active <= 100, f"Too many active neurons: {n_active}"

    # Check activation values in [0, 1]
    assert (new_state.activation >= 0).all()
    assert (new_state.activation <= 1).all()

    print("✅ NeuronInteraction test passed")


def test_output_decoder():
    """Test OutputDecoder module."""
    n_neurons = 512
    d_state = 128
    vocab_size = 1000

    decoder = OutputDecoder(
        n_neurons=n_neurons,
        d_state=d_state,
        vocab_size=vocab_size
    )

    # Create state
    state = NeuronState.create(n_neurons, d_state, torch.device('cpu'))

    # Activate some neurons
    active_indices = torch.tensor([0, 10, 20, 30, 40])
    state.activation[active_indices] = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])

    # Test decoder
    logits = decoder(state)

    # Check shape
    assert logits.shape == (vocab_size,)

    # Logits can be any value
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()

    print("✅ OutputDecoder test passed")


def test_sprout_brain_like():
    """Test full SPROUT_BrainLike model."""
    vocab_size = 1000
    n_neurons = 512
    d_state = 128
    batch_size = 2
    seq_len = 10

    model = create_brain_like_sprout(
        vocab_size=vocab_size,
        n_neurons=n_neurons,
        d_state=d_state,
        n_interaction_steps=3
    )

    # Test forward
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(tokens)

    # Check shape
    assert logits.shape == (batch_size, vocab_size)

    # Check no NaN/Inf
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()

    print("✅ SPROUT_BrainLike forward test passed")


def test_activation_analysis():
    """Test activation analysis functionality."""
    vocab_size = 1000
    n_neurons = 512
    d_state = 128
    seq_len = 10

    model = create_brain_like_sprout(
        vocab_size=vocab_size,
        n_neurons=n_neurons,
        d_state=d_state,
        n_interaction_steps=5
    )

    # Test analysis
    tokens = torch.randint(0, vocab_size, (1, seq_len))
    analysis = model.analyze_activation(tokens)

    # Check analysis structure
    assert 'n_steps' in analysis
    assert 'initial_active' in analysis
    assert 'final_active' in analysis
    assert 'activation_history' in analysis
    assert 'top_neurons_per_step' in analysis

    # Check values
    assert analysis['n_steps'] == 6  # 5 interaction steps + initial
    assert analysis['initial_active'] > 0
    assert analysis['final_active'] > 0

    # Check history
    history = analysis['activation_history']
    assert len(history) == 6

    # Check top neurons
    top_neurons = analysis['top_neurons_per_step']
    assert len(top_neurons) == 6

    for step_info in top_neurons:
        assert 'step' in step_info
        assert 'indices' in step_info
        assert 'values' in step_info
        assert len(step_info['indices']) == 10  # Top 10

    print("✅ Activation analysis test passed")


def test_sparse_activation_maintained():
    """Test that sparse activation is maintained throughout processing."""
    vocab_size = 1000
    n_neurons = 512
    d_state = 128
    seq_len = 10
    initial_sparsity = 64
    final_sparsity = 128

    model = create_brain_like_sprout(
        vocab_size=vocab_size,
        n_neurons=n_neurons,
        d_state=d_state,
        n_interaction_steps=5,
        initial_sparsity=initial_sparsity,
        final_sparsity=final_sparsity
    )

    tokens = torch.randint(0, vocab_size, (1, seq_len))
    analysis = model.analyze_activation(tokens)

    # Check initial sparsity
    initial_active = analysis['initial_active']
    assert initial_active == initial_sparsity, \
        f"Expected {initial_sparsity} initial active, got {initial_active}"

    # Check that final doesn't exceed max
    final_active = analysis['final_active']
    assert final_active <= final_sparsity, \
        f"Final active ({final_active}) exceeds max ({final_sparsity})"

    # Check all intermediate steps
    for i, activation in enumerate(analysis['activation_history']):
        n_active = (activation > 0.01).sum().item()
        if i == 0:
            assert n_active == initial_sparsity
        else:
            assert n_active <= final_sparsity, \
                f"Step {i}: {n_active} active exceeds {final_sparsity}"

    print("✅ Sparse activation maintained test passed")


def test_different_inputs_different_patterns():
    """Test that different inputs produce different activation patterns."""
    vocab_size = 1000
    n_neurons = 512
    d_state = 128
    seq_len = 10

    model = create_brain_like_sprout(
        vocab_size=vocab_size,
        n_neurons=n_neurons,
        d_state=d_state,
        n_interaction_steps=3
    )

    # Two different inputs
    tokens1 = torch.randint(0, vocab_size, (1, seq_len))
    tokens2 = torch.randint(0, vocab_size, (1, seq_len))

    # Make sure they're different
    while torch.equal(tokens1, tokens2):
        tokens2 = torch.randint(0, vocab_size, (1, seq_len))

    # Get activation patterns
    with torch.no_grad():
        analysis1 = model.analyze_activation(tokens1)
        analysis2 = model.analyze_activation(tokens2)

    act1 = analysis1['activation_history'][-1]
    act2 = analysis2['activation_history'][-1]

    # Patterns should be different
    # (Not exactly the same, allowing for some randomness)
    assert not torch.allclose(act1, act2, atol=1e-6), \
        "Different inputs produced identical patterns"

    print("✅ Different inputs produce different patterns test passed")


def test_gradient_flow():
    """Test that gradients flow through the model."""
    vocab_size = 1000
    n_neurons = 512
    d_state = 128
    batch_size = 2
    seq_len = 10

    model = create_brain_like_sprout(
        vocab_size=vocab_size,
        n_neurons=n_neurons,
        d_state=d_state,
        n_interaction_steps=3
    )

    # Forward pass
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(tokens)

    # Dummy loss
    target = torch.randint(0, vocab_size, (batch_size,))
    loss = torch.nn.functional.cross_entropy(logits, target)

    # Backward
    loss.backward()

    # Check that gradients exist
    has_grad = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "No gradients found in model parameters"

    print("✅ Gradient flow test passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("BRAIN-LIKE SPROUT TESTS")
    print("="*70 + "\n")

    test_global_neuron_pool()
    test_neuron_state()
    test_input_to_activation()
    test_neuron_interaction()
    test_output_decoder()
    test_sprout_brain_like()
    test_activation_analysis()
    test_sparse_activation_maintained()
    test_different_inputs_different_patterns()
    test_gradient_flow()

    print("\n" + "="*70)
    print("ALL TESTS PASSED! ✅")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_all_tests()
