"""
转换 Worker 线程

在后台线程中执行音频转换，避免阻塞 UI
"""

import os
from typing import Dict, Optional

from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker


class ConversionWorker(QThread):
    """
    音频转换 Worker
    
    Signals:
        progress(int, str): 进度更新 (0-100, 描述文本)
        finished(str): 转换完成，输出文件路径
        error(str): 转换失败，错误信息
        log(str): 日志消息
    """
    
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    log = pyqtSignal(str)
    
    def __init__(
        self,
        input_file: str,
        output_file: str,
        engine: str,
        params: Dict,
        parent=None
    ):
        super().__init__(parent)
        
        self._input_file = input_file
        self._output_file = output_file
        self._engine = engine
        self._params = params
        
        self._is_cancelled = False
        self._mutex = QMutex()
    
    def run(self) -> None:
        """执行转换"""
        try:
            self.log.emit(f"开始转换: {self._engine}")
            self.log.emit(f"输入文件: {self._input_file}")
            self.log.emit(f"输出文件: {self._output_file}")
            
            # 更新参数
            params = self._params.copy()
            params['output_path'] = self._output_file
            
            # 根据引擎选择转换器
            if self._engine == "rvc":
                self._convert_rvc()
            else:
                self._convert_sovits()
            
            if self._is_cancelled:
                self.error.emit("转换已取消")
                return
            
            self.progress.emit(100, "转换完成")
            self.log.emit(f"转换完成: {self._output_file}")
            self.finished.emit(self._output_file)
            
        except Exception as e:
            self.error.emit(str(e))
    
    def _convert_rvc(self) -> None:
        """RVC 转换"""
        try:
            from src.voice_converters import RVCConverter
            
            self.progress.emit(5, "加载 RVC 模型...")
            self.log.emit(f"模型: {self._params.get('model_path')}")
            self.log.emit(f"索引: {self._params.get('index_path', '无')}")
            
            converter = RVCConverter()
            
            # 加载模型
            model_path = self._params['model_path']
            index_path = self._params.get('index_path', '')
            
            converter.load_model(
                model_path,
                index_path=index_path if index_path else None
            )
            
            self.progress.emit(30, "模型加载完成")
            
            # 转换
            self.progress.emit(40, "开始转换...")
            self.log.emit(f"参数: f0={self._params['f0_up_key']}, 算法={self._params['f0_algo']}")
            
            # 读取输入音频
            import numpy as np
            import soundfile as sf
            
            audio, sr = sf.read(self._input_file, dtype='float32')
            self.log.emit(f"音频: {len(audio)/sr:.2f}s, {sr}Hz")
            
            # 转换
            result = converter.convert(
                audio,
                sr,
                f0_up_key=self._params['f0_up_key'],
                f0_algo=self._params['f0_algo'],
                index_ratio=self._params['index_ratio'],
                rms_mix=self._params['rms_mix'],
                protect=self._params['protect'],
                output_path=self._params['output_path']
            )
            
            self.progress.emit(90, "保存结果...")
            
            # 导出
            sf.write(
                self._params['output_path'],
                result.audio,
                result.sampling_rate
            )
            
            # 卸载模型
            converter.unload()
            
        except Exception as e:
            self.error.emit(f"RVC 转换失败: {str(e)}")
            raise
    
    def _convert_sovits(self) -> None:
        """SoVITS 转换"""
        try:
            from src.voice_converters import SoVITSConverter
            
            self.progress.emit(5, "加载 SoVITS 模型...")
            self.log.emit(f"模型: {self._params.get('model_path')}")
            self.log.emit(f"配置: {self._params.get('config_path', '无')}")
            
            converter = SoVITSConverter()
            
            # 加载模型
            model_path = self._params['model_path']
            config_path = self._params.get('config_path', '')
            
            converter.load_model(
                model_path,
                config_path=config_path if config_path else None
            )
            
            self.progress.emit(30, "模型加载完成")
            
            # 转换
            self.progress.emit(40, "开始转换...")
            self.log.emit(f"参数: f0={self._params['f0_up_key']}, 算法={self._params['f0_algo']}")
            
            # 读取输入音频
            import numpy as np
            import soundfile as sf
            
            audio, sr = sf.read(self._input_file, dtype='float32')
            self.log.emit(f"音频: {len(audio)/sr:.2f}s, {sr}Hz")
            
            # 转换
            result = converter.convert(
                audio,
                sr,
                f0_up_key=self._params['f0_up_key'],
                f0_algo=self._params['f0_algo'],
                diff=1,  # 启用扩散
                output_path=self._params['output_path']
            )
            
            self.progress.emit(90, "保存结果...")
            
            # 导出
            sf.write(
                self._params['output_path'],
                result.audio,
                result.sampling_rate
            )
            
            # 卸载模型
            converter.unload()
            
        except Exception as e:
            self.error.emit(f"SoVITS 转换失败: {str(e)}")
            raise
    
    def cancel(self) -> None:
        """取消转换"""
        with QMutexLocker(self._mutex):
            self._is_cancelled = True
