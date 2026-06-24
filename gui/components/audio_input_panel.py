"""
音频输入面板

功能:
- 拖拽或选择音频文件
- 显示音频波形预览
- 播放/暂停控制
"""

import os
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QGroupBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPixmap, QPainter, QColor, QPen


class WaveformWidget(QLabel):
    """波形显示组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.setMaximumHeight(150)
        self.setStyleSheet("background-color: #1e1e1e; border: 1px solid #3c3c3c;")
        self._waveform_data = None
    
    def set_waveform(self, data):
        """设置波形数据"""
        self._waveform_data = data
        self.update()
    
    def paintEvent(self, event):
        """绘制波形"""
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        if self._waveform_data is None:
            # 绘制占位文字
            painter.setPen(QColor("#757575"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "拖拽音频文件到此处")
            return
        
        # 获取绘图区域
        rect = self.rect()
        width = rect.width()
        height = rect.height()
        
        # 设置绘图样式
        painter.setPen(QPen(QColor("#4fc3f7"), 2))
        
        # 绘制波形
        data = self._waveform_data
        if len(data) > 0:
            # 缩放到控件宽度
            step = max(1, len(data) // width)
            
            center_y = height // 2
            max_amplitude = height // 2 - 10
            
            for x in range(0, width):
                idx = min(x * step, len(data) - 1)
                amplitude = abs(data[idx]) * max_amplitude
                
                painter.drawLine(x, center_y - amplitude, x, center_y + amplitude)


class AudioInputPanel(QWidget):
    """
    音频输入面板
    
    Signals:
        file_selected(str): 文件选择信号
        play_requested(str): 播放请求信号
    """
    
    file_selected = pyqtSignal(str)
    play_requested = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._current_file: Optional[str] = None
        self._waveform_data = None
        
        self._init_ui()
        self._init_drag_drop()
    
    def _init_ui(self) -> None:
        """初始化 UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 面板标题
        group = QGroupBox("音频输入")
        group_layout = QVBoxLayout()
        
        # 波形显示
        self.waveform_widget = WaveformWidget()
        group_layout.addWidget(self.waveform_widget)
        
        # 文件信息
        self.file_label = QLabel("未选择文件")
        self.file_label.setStyleSheet("color: #757575; padding: 5px;")
        group_layout.addWidget(self.file_label)
        
        # 按钮区域
        btn_layout = QHBoxLayout()
        
        self.open_btn = QPushButton("选择文件")
        self.open_btn.clicked.connect(self._on_select_file)
        btn_layout.addWidget(self.open_btn)
        
        self.play_btn = QPushButton("播放")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self._on_play)
        btn_layout.addWidget(self.play_btn)
        
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        btn_layout.addWidget(self.stop_btn)
        
        group_layout.addLayout(btn_layout)
        group.setLayout(group_layout)
        layout.addWidget(group)
        
        # 保存组件引用
        self._group = group
        self._btn_layout = btn_layout
    
    def _init_drag_drop(self) -> None:
        """初始化拖放功能"""
        self.setAcceptDrops(True)
    
    def set_file(self, file_path: str) -> None:
        """设置音频文件"""
        if not os.path.exists(file_path):
            return
        
        self._current_file = file_path
        
        # 更新显示
        filename = os.path.basename(file_path)
        self.file_label.setText(filename)
        self.file_label.setStyleSheet("color: #4fc3f7; padding: 5px;")
        
        # 启用播放按钮
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # 加载波形
        self._load_waveform(file_path)
        
        # 发送信号
        self.file_selected.emit(file_path)
    
    def _load_waveform(self, file_path: str) -> None:
        """加载波形数据"""
        try:
            import numpy as np
            import soundfile as sf
            
            # 读取音频
            audio, sr = sf.read(file_path, dtype='float32')
            
            # 如果是立体声，转为单声道
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # 简化波形数据（下采样到 1000 点）
            if len(audio) > 1000:
                step = len(audio) // 1000
                self._waveform_data = audio[::step]
            else:
                self._waveform_data = audio
            
            self.waveform_widget.set_waveform(self._waveform_data)
            
        except Exception as e:
            # 加载失败，显示空波形
            self._waveform_data = None
            self.waveform_widget.set_waveform(None)
    
    # ==================== 事件处理 ====================
    
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """拖拽进入"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._group.setStyleSheet("border: 2px dashed #4fc3f7;")
    
    def dragLeaveEvent(self, event) -> None:
        """拖拽离开"""
        self._group.setStyleSheet("")
    
    def dropEvent(self, event: QDropEvent) -> None:
        """放下文件"""
        self._group.setStyleSheet("")
        
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            
            # 检查是否是音频文件
            ext = os.path.splitext(file_path)[1].lower()
            audio_exts = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
            
            if ext in audio_exts:
                self.set_file(file_path)
            else:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "不支持的文件", f"不支持的音频格式: {ext}")
    
    def _on_select_file(self) -> None:
        """选择文件按钮"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择音频文件",
            "",
            "音频文件 (*.wav *.mp3 *.flac *.ogg *.m4a);;所有文件 (*.*)"
        )
        
        if file_path:
            self.set_file(file_path)
    
    def _on_play(self) -> None:
        """播放按钮"""
        if self._current_file:
            self.play_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.play_requested.emit(self._current_file)
    
    def _on_stop(self) -> None:
        """停止按钮"""
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        # TODO: 发送停止播放信号
