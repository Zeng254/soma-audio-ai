"""
Audio Pipeline - Audio processing pipeline
Supports chaining multiple processing nodes
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any, Dict
from enum import Enum
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class Node types(Enum):
    """Node types"""
    SEPARATOR = "separator"
    EFFECT = "effect"
    CONVERTER = "converter"
    FILTER = "filter"
    CUSTOM = "custom"


@dataclass
class Pipeline node:
    """
    Pipeline node
    
    Represents a processing step in the pipeline
    """
    name: str                                    # NodeName
    node_type: Node types                          # Node types
    process_fn: Callable[[np.ndarray, int], tuple]  # ProcessFunction
    params: Dict[str, Any] = field(default_factory=dict)  # ProcessParameter
    enabled: bool = True                         # Whether enabled
    bypass: bool = False                         # Whether bypassed
    
    def execute(self, audio: np.ndarray, sample_rate: int) -> tuple:
        """
        ExecuteNodeProcess
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
            
        Returns:
            (output_audio, sample_rate)
        """
        if not self.enabled or self.bypass:
            return audio, sample_rate
        
        return self.process_fn(audio, sample_rate, **self.params)


@dataclass
class PipelineResult:
    """Pipeline execution result"""
    audio: np.ndarray
    sample_rate: int
    duration: float
    nodes_executed: List[str]
    node_times: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None


class AudioPipeline:
    """
    Audio processing pipeline
    
    Supports chaining multiple audio processing nodes,
    Each node can be a separator, effect processor, converter, etc.
    
    Example:
        pipeline = AudioPipeline(sample_rate=44100)
        pipeline.add_separator("demucs", DemucsSeparator())
        pipeline.add_effect("eq", Equalizer(), preset="pop")
        pipeline.add_effect("reverb", Reverb(), room_size=0.7)
        result = pipeline.execute(audio)
    """
    
    def __init__(self, name: str = "pipeline", sample_rate: int = 44100):
        """
        Initialize pipeline
        
        Args:
            name: Pipeline name
            sample_rate: Default sample rate
        """
        self.name = name
        self.default_sample_rate = sample_rate
        self.nodes: List[Pipeline node] = []
        self._node_times: Dict[str, float] = {}
    
    def add_node(
        self,
        name: str,
        node_type: Node types,
        process_fn: Callable,
        params: Optional[Dict] = None,
        enabled: bool = True,
    ) -> "AudioPipeline":
        """
        AddProcessNode
        
        Args:
            name: NodeName
            node_type: Node types
            process_fn: ProcessFunction
            params: ProcessParameter
            enabled: Whether enabled
            
        Returns:
            self
        """
        node = Pipeline node(
            name=name,
            node_type=node_type,
            process_fn=process_fn,
            params=params or {},
            enabled=enabled,
        )
        self.nodes.append(node)
        return self
    
    def add_separator(
        self,
        name: str,
        separator: Any,
        **params
    ) -> "AudioPipeline":
        """
        AddSeparatorNode
        
        Args:
            name: NodeName
            separator: SeparatorInstance
            **params: SeparationParameter
            
        Returns:
            self
        """
        def process(audio, sr, **kwargs):
            result = separator.separate_array(audio, sr, **kwargs)
            return result.vocals if result.vocals is not None else audio, sr
        
        return self.add_node(
            name=name,
            node_type=Node types.SEPARATOR,
            process_fn=process,
            params=params,
        )
    
    def add_effect(
        self,
        name: str,
        effect: Any,
        **params
    ) -> "AudioPipeline":
        """
        Add effect processor node
        
        Args:
            name: NodeName
            effect: Effect processor instance
            **params: EffectParameter
            
        Returns:
            self
        """
        def process(audio, sr, **kwargs):
            result = effect.process(audio, sr, **kwargs)
            return result.audio, sr
        
        return self.add_node(
            name=name,
            node_type=Node types.EFFECT,
            process_fn=process,
            params=params,
        )
    
    def add_filter(
        self,
        name: str,
        filter_fn: Callable[[np.ndarray, int, dict], np.ndarray],
        params: Optional[dict] = None,
    ) -> "AudioPipeline":
        """
        Add custom filter node
        
        Args:
            name: NodeName
            filter_fn: FilterFunction
            params: FilterParameter
            
        Returns:
            self
        """
        def process(audio, sr, **kwargs):
            return filter_fn(audio, sr, kwargs), sr
        
        return self.add_node(
            name=name,
            node_type=Node types.FILTER,
            process_fn=process,
            params=params,
        )
    
    def add_custom(
        self,
        name: str,
        process_fn: Callable[[np.ndarray, int], tuple],
        params: Optional[dict] = None,
    ) -> "AudioPipeline":
        """
        Add custom processing node
        
        Args:
            name: NodeName
            process_fn: ProcessFunction (audio, sample_rate) -> (output, sample_rate)
            params: ProcessParameter
            
        Returns:
            self
        """
        def wrapped_process(audio, sr, **kwargs):
            return process_fn(audio, sr)
        
        return self.add_node(
            name=name,
            node_type=Node types.CUSTOM,
            process_fn=wrapped_process,
            params=params,
        )
    
    def remove_node(self, name: str) -> bool:
        """RemoveNode"""
        for i, node in enumerate(self.nodes):
            if node.name == name:
                self.nodes.pop(i)
                return True
        return False
    
    def get_node(self, name: str) -> Optional[Pipeline node]:
        """GetNode"""
        for node in self.nodes:
            if node.name == name:
                return node
        return None
    
    def set_bypass(self, name: str, bypass: bool = True):
        """Set node bypass"""
        node = self.get_node(name)
        if node:
            node.bypass = bypass
    
    def reorder_nodes(self, names: List[str]):
        """
        Re-sort nodes
        
        Args:
            names: New node name order
        """
        name_to_node = {node.name: node for node in self.nodes}
        self.nodes = [name_to_node[name] for name in names if name in name_to_node]
    
    def execute(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
        return_tracks: bool = False,
    ) -> PipelineResult:
        """
        Execute pipeline
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
            return_tracks: Whether to return all tracks
            
        Returns:
            PipelineResult: Execution result
        """
        if sample_rate is None:
            sample_rate = self.default_sample_rate
        
        start_time = time.time()
        current_audio = audio
        current_sr = sample_rate
        nodes_executed = []
        node_times = {}
        
        # Record separation result
        separation_results = {}
        
        for node in self.nodes:
            if not node.enabled:
                continue
            
            node_start = time.time()
            
            try:
                if node.node_type == Node types.SEPARATOR and return_tracks:
                    # Separator node, save all tracks
                    result_audio, result_sr = node.execute(current_audio, current_sr)
                    separation_results[node.name] = result_audio
                else:
                    current_audio, current_sr = node.execute(current_audio, current_sr)
                
                nodes_executed.append(node.name)
                node_times[node.name] = time.time() - node_start
                
            except Exception as e:
                logger.error(f"Error in node {node.name}: {e}", exc_info=True)
                node_times[node.name] = time.time() - node_start
        
        total_time = time.time() - start_time
        
        return PipelineResult(
            audio=current_audio,
            sample_rate=current_sr,
            duration=total_time,
            nodes_executed=nodes_executed,
            node_times=node_times,
            metadata={"separation_results": separation_results} if separation_results else None,
        )
    
    def execute_file(
        self,
        input_path: str,
        output_path: str,
        sample_rate: Optional[int] = None,
    ) -> bool:
        """
        ProcessAudioFile
        
        Args:
            input_path: Input file
            output_path: Output file
            sample_rate: Sample rate
            
        Returns:
            bool: Whether successful
        """
        from src.utils.audio_io import AudioLoader, AudioSaver
        
        loader = AudioLoader()
        saver = AudioSaver()
        
        audio, sr = loader.load(input_path)
        if sample_rate:
            sr = sample_rate
        
        result = self.execute(audio, sr)
        
        return saver.save(result.audio, output_path, sr)
    
    def clear(self):
        """Clear all nodes"""
        self.nodes.clear()
        self._node_times.clear()
    
    def __len__(self) -> int:
        """NodeCount"""
        return len(self.nodes)
    
    def __repr__(self) -> str:
        return f"AudioPipeline(name={self.name}, nodes={len(self.nodes)})"
    
    def summary(self) -> str:
        """Generate pipeline summary"""
        lines = [f"Pipeline: {self.name}", f"Sample Rate: {self.default_sample_rate}Hz", "", "Nodes:"]
        
        for i, node in enumerate(self.nodes, 1):
            status = "✓" if node.enabled else "✗"
            bypass = " [BYPASS]" if node.bypass else ""
            lines.append(f"  {i}. [{status}] {node.name} ({node.node_type.value}){bypass}")
        
        return "\n".join(lines)


class PipelineBuilder:
    """
    Pipeline builder
    
    Provides fluent API to build pipeline
    """
    
    def __init__(self, name: str = "pipeline", sample_rate: int = 44100):
        self.pipeline = AudioPipeline(name=name, sample_rate=sample_rate)
    
    def with_separator(self, name: str, separator: Any, **params) -> "PipelineBuilder":
        """AddSeparator"""
        self.pipeline.add_separator(name, separator, **params)
        return self
    
    def with_effect(self, name: str, effect: Any, **params) -> "PipelineBuilder":
        """Add effect processor"""
        self.pipeline.add_effect(name, effect, **params)
        return self
    
    def with_filter(self, name: str, filter_fn: Callable, params: Optional[dict] = None) -> "PipelineBuilder":
        """Add filter processor"""
        self.pipeline.add_filter(name, filter_fn, params)
        return self
    
    def with_custom(self, name: str, process_fn: Callable, params: Optional[dict] = None) -> "PipelineBuilder":
        """Add custom node"""
        self.pipeline.add_custom(name, process_fn, params)
        return self
    
    def build(self) -> AudioPipeline:
        """Build pipeline"""
        return self.pipeline
