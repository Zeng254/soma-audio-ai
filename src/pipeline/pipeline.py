"""
Audio Pipeline - 音频处理流水线
支持链式调用多个处理节点
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any, Dict
from enum import Enum
import numpy as np
import time


class NodeType(Enum):
    """节点类型"""
    SEPARATOR = "separator"
    EFFECT = "effect"
    CONVERTER = "converter"
    FILTER = "filter"
    CUSTOM = "custom"


@dataclass
class PipelineNode:
    """
    流水线节点
    
    代表流水线中的一个处理步骤
    """
    name: str                                    # 节点名称
    node_type: NodeType                          # 节点类型
    process_fn: Callable[[np.ndarray, int], tuple]  # 处理函数
    params: Dict[str, Any] = field(default_factory=dict)  # 处理参数
    enabled: bool = True                         # 是否启用
    bypass: bool = False                         # 是否旁路
    
    def execute(self, audio: np.ndarray, sample_rate: int) -> tuple:
        """
        执行节点处理
        
        Args:
            audio: 输入音频
            sample_rate: 采样率
            
        Returns:
            (output_audio, sample_rate)
        """
        if not self.enabled or self.bypass:
            return audio, sample_rate
        
        return self.process_fn(audio, sample_rate, **self.params)


@dataclass
class PipelineResult:
    """流水线执行结果"""
    audio: np.ndarray
    sample_rate: int
    duration: float
    nodes_executed: List[str]
    node_times: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None


class AudioPipeline:
    """
    音频处理流水线
    
    支持链式调用多个音频处理节点，
    每个节点可以是分离器、效果器、转换器等。
    
    示例:
        pipeline = AudioPipeline(sample_rate=44100)
        pipeline.add_separator("demucs", DemucsSeparator())
        pipeline.add_effect("eq", Equalizer(), preset="pop")
        pipeline.add_effect("reverb", Reverb(), room_size=0.7)
        result = pipeline.execute(audio)
    """
    
    def __init__(self, name: str = "pipeline", sample_rate: int = 44100):
        """
        初始化流水线
        
        Args:
            name: 流水线名称
            sample_rate: 默认采样率
        """
        self.name = name
        self.default_sample_rate = sample_rate
        self.nodes: List[PipelineNode] = []
        self._node_times: Dict[str, float] = {}
    
    def add_node(
        self,
        name: str,
        node_type: NodeType,
        process_fn: Callable,
        params: Optional[Dict] = None,
        enabled: bool = True,
    ) -> "AudioPipeline":
        """
        添加处理节点
        
        Args:
            name: 节点名称
            node_type: 节点类型
            process_fn: 处理函数
            params: 处理参数
            enabled: 是否启用
            
        Returns:
            self
        """
        node = PipelineNode(
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
        添加分离器节点
        
        Args:
            name: 节点名称
            separator: 分离器实例
            **params: 分离参数
            
        Returns:
            self
        """
        def process(audio, sr, **kwargs):
            result = separator.separate_array(audio, sr, **kwargs)
            return result.vocals if result.vocals is not None else audio, sr
        
        return self.add_node(
            name=name,
            node_type=NodeType.SEPARATOR,
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
        添加效果器节点
        
        Args:
            name: 节点名称
            effect: 效果器实例
            **params: 效果参数
            
        Returns:
            self
        """
        def process(audio, sr, **kwargs):
            result = effect.process(audio, sr, **kwargs)
            return result.audio, sr
        
        return self.add_node(
            name=name,
            node_type=NodeType.EFFECT,
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
        添加自定义过滤器节点
        
        Args:
            name: 节点名称
            filter_fn: 过滤函数
            params: 过滤参数
            
        Returns:
            self
        """
        def process(audio, sr, **kwargs):
            return filter_fn(audio, sr, kwargs), sr
        
        return self.add_node(
            name=name,
            node_type=NodeType.FILTER,
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
        添加自定义处理节点
        
        Args:
            name: 节点名称
            process_fn: 处理函数 (audio, sample_rate) -> (output, sample_rate)
            params: 处理参数
            
        Returns:
            self
        """
        def wrapped_process(audio, sr, **kwargs):
            return process_fn(audio, sr)
        
        return self.add_node(
            name=name,
            node_type=NodeType.CUSTOM,
            process_fn=wrapped_process,
            params=params,
        )
    
    def remove_node(self, name: str) -> bool:
        """移除节点"""
        for i, node in enumerate(self.nodes):
            if node.name == name:
                self.nodes.pop(i)
                return True
        return False
    
    def get_node(self, name: str) -> Optional[PipelineNode]:
        """获取节点"""
        for node in self.nodes:
            if node.name == name:
                return node
        return None
    
    def set_bypass(self, name: str, bypass: bool = True):
        """设置节点旁路"""
        node = self.get_node(name)
        if node:
            node.bypass = bypass
    
    def reorder_nodes(self, names: List[str]):
        """
        重新排序节点
        
        Args:
            names: 新的节点名称顺序
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
        执行流水线
        
        Args:
            audio: 输入音频
            sample_rate: 采样率
            return_tracks: 是否返回所有音轨
            
        Returns:
            PipelineResult: 执行结果
        """
        if sample_rate is None:
            sample_rate = self.default_sample_rate
        
        start_time = time.time()
        current_audio = audio
        current_sr = sample_rate
        nodes_executed = []
        node_times = {}
        
        # 记录分离结果
        separation_results = {}
        
        for node in self.nodes:
            if not node.enabled:
                continue
            
            node_start = time.time()
            
            try:
                if node.node_type == NodeType.SEPARATOR and return_tracks:
                    # 分离器节点，保存所有音轨
                    result_audio, result_sr = node.execute(current_audio, current_sr)
                    separation_results[node.name] = result_audio
                else:
                    current_audio, current_sr = node.execute(current_audio, current_sr)
                
                nodes_executed.append(node.name)
                node_times[node.name] = time.time() - node_start
                
            except Exception as e:
                print(f"Error in node {node.name}: {e}")
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
        处理音频文件
        
        Args:
            input_path: 输入文件
            output_path: 输出文件
            sample_rate: 采样率
            
        Returns:
            bool: 是否成功
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
        """清空所有节点"""
        self.nodes.clear()
        self._node_times.clear()
    
    def __len__(self) -> int:
        """节点数量"""
        return len(self.nodes)
    
    def __repr__(self) -> str:
        return f"AudioPipeline(name={self.name}, nodes={len(self.nodes)})"
    
    def summary(self) -> str:
        """生成流水线摘要"""
        lines = [f"Pipeline: {self.name}", f"Sample Rate: {self.default_sample_rate}Hz", "", "Nodes:"]
        
        for i, node in enumerate(self.nodes, 1):
            status = "✓" if node.enabled else "✗"
            bypass = " [BYPASS]" if node.bypass else ""
            lines.append(f"  {i}. [{status}] {node.name} ({node.node_type.value}){bypass}")
        
        return "\n".join(lines)


class PipelineBuilder:
    """
    流水线构建器
    
    提供流畅的 API 来构建流水线
    """
    
    def __init__(self, name: str = "pipeline", sample_rate: int = 44100):
        self.pipeline = AudioPipeline(name=name, sample_rate=sample_rate)
    
    def with_separator(self, name: str, separator: Any, **params) -> "PipelineBuilder":
        """添加分离器"""
        self.pipeline.add_separator(name, separator, **params)
        return self
    
    def with_effect(self, name: str, effect: Any, **params) -> "PipelineBuilder":
        """添加效果器"""
        self.pipeline.add_effect(name, effect, **params)
        return self
    
    def with_filter(self, name: str, filter_fn: Callable, params: Optional[dict] = None) -> "PipelineBuilder":
        """添加过滤器"""
        self.pipeline.add_filter(name, filter_fn, params)
        return self
    
    def with_custom(self, name: str, process_fn: Callable, params: Optional[dict] = None) -> "PipelineBuilder":
        """添加自定义节点"""
        self.pipeline.add_custom(name, process_fn, params)
        return self
    
    def build(self) -> AudioPipeline:
        """构建流水线"""
        return self.pipeline
