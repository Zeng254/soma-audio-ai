"""
SOMA Pipeline Module
Processing pipeline module - Supports chaining multiple processing nodes
"""

from .pipeline import AudioPipeline, Pipeline node, PipelineBuilder

__all__ = [
    "AudioPipeline",
    "Pipeline node",
    "PipelineBuilder",
]
