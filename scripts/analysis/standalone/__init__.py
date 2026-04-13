"""
Standalone Analysis Scripts
============================
CLI tools for specific analysis tasks.

Scripts are designed to run independently. Import specific modules directly:
    from scripts.analysis.standalone.routing_analysis import GenerationRoutingAnalyzer
    from scripts.analysis.standalone.neuron_suppression_experiment_jax import ...

No top-level imports to avoid torch/jax dependency conflicts.
"""
