"""
状态栏组件

功能:
- 显示系统状态
- GPU/CPU 状态
- 内存使用情况
"""

from PyQt6.QtWidgets import QStatusBar, QLabel, QProgressBar
from PyQt6.QtCore import QTimer


class StatusBar(QStatusBar):
    """状态栏组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._init_ui()
        self._start_monitoring()
    
    def _init_ui(self) -> None:
        """初始化 UI"""
        # 状态标签
        self.status_label = QLabel("就绪")
        self.addWidget(self.status_label)
        
        # 分隔符
        self.addPermanentWidget(QLabel("|"))
        
        # GPU 标签
        self.gpu_label = QLabel("GPU: 未检测")
        self.addPermanentWidget(self.gpu_label)
        
        # 分隔符
        self.addPermanentWidget(QLabel("|"))
        
        # 内存标签
        self.memory_label = QLabel("内存: -")
        self.addPermanentWidget(self.memory_label)
        
        # 分隔符
        self.addPermanentWidget(QLabel("|"))
        
        # 版本标签
        self.version_label = QLabel("v0.1.0")
        self.addPermanentWidget(self.version_label)
    
    def _start_monitoring(self) -> None:
        """开始状态监控"""
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_status)
        self._timer.start(5000)  # 每 5 秒更新
    
    def _update_status(self) -> None:
        """更新状态"""
        try:
            # 检测 GPU
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    self.gpu_label.setText(f"GPU: {gpu_name[:20]} ({gpu_memory:.1f}/{gpu_total:.1f}GB)")
                else:
                    self.gpu_label.setText("GPU: 不可用")
            except:
                self.gpu_label.setText("GPU: 未检测")
            
            # 检测内存
            try:
                import psutil
                memory = psutil.virtual_memory()
                self.memory_label.setText(
                    f"内存: {memory.percent:.0f}%"
                )
            except:
                self.memory_label.setText("内存: -")
                
        except Exception:
            pass
    
    def set_status(self, message: str, timeout: int = 0) -> None:
        """设置状态消息"""
        self.status_label.setText(message)
        if timeout > 0:
            QTimer.singleShot(timeout, lambda: self.status_label.setText("就绪"))
    
    def set_ready(self) -> None:
        """设置为就绪状态"""
        self.status_label.setText("就绪")
        self.status_label.setStyleSheet("")
