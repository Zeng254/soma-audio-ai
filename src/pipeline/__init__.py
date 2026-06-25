"""
SOMA Pipeline Module
Processing pipeline module - Supports chaining multiple processing nodes
"""

from .pipeline import AudioPipeline, PipelineNode, PipelineBuilder

__all__ = [
    "AudioPipeline",
    "PipelineNode",
    "PipelineBuilder",
]
