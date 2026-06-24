"""
输出面板

功能:
- 显示转换结果
- 预览输出音频
- 导出音频文件
"""

import os
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QGroupBox, QProgressBar,
    QTextEdit
)
from PyQt6.QtCore import Qt, pyqtSignal


class OutputPanel(QWidget):
    """
    输出面板
    
    Signals:
        play_output_requested(str): 播放输出文件
        save_output_requested(str): 保存输出文件
    """
    
    play_requested = pyqtSignal(str)
    save_requested = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._output_file: Optional[str] = None
        self._init_ui()
    
    def _init_ui(self) -> None:
        """初始化 UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 输出预览
        preview_group = QGroupBox("转换结果")
        preview_layout = QVBoxLayout()
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.hide()
        preview_layout.addWidget(self.progress_bar)
        
        # 状态文本
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setReadOnly(True)
        self.status_text.setPlaceholderText("转换状态将显示在这里...")
        self.status_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #4fc3f7;
                border: 1px solid #3c3c3c;
                font-family: Consolas, monospace;
                font-size: 12px;
            }
        """)
        preview_layout.addWidget(self.status_text)
        
        # 输出信息
        self.output_label = QLabel("未生成输出文件")
        self.output_label.setStyleSheet("color: #757575; padding: 5px;")
        preview_layout.addWidget(self.output_label)
        
        # 按钮区域
        btn_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("播放")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self._on_play)
        btn_layout.addWidget(self.play_btn)
        
        self.save_btn = QPushButton("导出")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._on_save)
        btn_layout.addWidget(self.save_btn)
        
        preview_layout.addLayout(btn_layout)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # 按钮区域
        btn_area = QHBoxLayout()
        
        self._preview_group = preview_group
        self._btn_layout = btn_layout
    
    def set_progress(self, value: int, text: str = "") -> None:
        """设置进度"""
        self.progress_bar.setValue(value)
        if text:
            self.status_text.append(text)
    
    def show_progress(self, show: bool = True) -> None:
        """显示/隐藏进度条"""
        if show:
            self.progress_bar.show()
            self.progress_bar.setValue(0)
        else:
            self.progress_bar.hide()
    
    def set_output(self, file_path: str) -> None:
        """设置输出文件"""
        self._output_file = file_path
        
        # 更新显示
        filename = os.path.basename(file_path)
        self.output_label.setText(filename)
        self.output_label.setStyleSheet("color: #4fc3f7; padding: 5px;")
        
        # 启用按钮
        self.play_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
    
    def clear_output(self) -> None:
        """清除输出"""
        self._output_file = None
        self.output_label.setText("未生成输出文件")
        self.output_label.setStyleSheet("color: #757575; padding: 5px;")
        self.play_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.status_text.clear()
        self.progress_bar.hide()
    
    def log(self, message: str) -> None:
        """添加日志"""
        self.status_text.append(message)
        # 滚动到底部
        scrollbar = self.status_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _on_play(self) -> None:
        """播放按钮"""
        if self._output_file:
            self.play_requested.emit(self._output_file)
    
    def _on_save(self) -> None:
        """导出按钮"""
        if self._output_file:
            default_name = os.path.basename(self._output_file)
            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "导出音频文件",
                default_name,
                "音频文件 (*.wav *.mp3 *.flac);;所有文件 (*.*)"
            )
            
            if save_path:
                import shutil
                shutil.copy2(self._output_file, save_path)
                self.save_requested.emit(save_path)
