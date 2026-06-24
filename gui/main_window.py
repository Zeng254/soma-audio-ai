"""
SOMA GUI - 主窗口实现

主要界面组件:
- 左侧: 音频输入区 (拖拽/选择文件、波形预览)
- 中间: 模型选择 + 参数调节面板
- 右侧: 输出预览 + 导出按钮
- 底部: 状态栏 + 转换进度条
"""

import os
import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QMainWindow, QPushButton, QLabel, QFileDialog,
    QMessageBox, QProgressBar, QStatusBar, QMenuBar,
    QMenu, QStyleFactory, QSplitter, QGroupBox,
    QComboBox, QSlider, QCheckBox, QSpinBox,
    QDoubleSpinBox, QListWidget, QListWidgetItem,
    QApplication
)
from PyQt6.QtCore import Qt, QThread, QSize, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QIcon, QDragEnterEvent, QDropEvent

# 导入组件
from .components import AudioInputPanel, ModelConfigPanel, OutputPanel, StatusBar
from .workers import ConversionWorker


class MainWindow(QMainWindow):
    """
    SOMA 主窗口
    
    Signals:
        conversion_started: 转换开始
        conversion_finished: 转换完成
        conversion_error: 转换错误
    """
    
    # 信号定义
    conversion_started = pyqtSignal()
    conversion_finished = pyqtSignal(str)  # 输出文件路径
    conversion_error = pyqtSignal(str)  # 错误信息
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        # 窗口属性
        self.setWindowTitle("SOMA 智能音频工作站")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        # 状态变量
        self._input_file: Optional[str] = None
        self._output_file: Optional[str] = None
        self._conversion_worker: Optional[ConversionWorker] = None
        self._current_engine: str = "rvc"  # rvc 或 sovits
        
        # 初始化 UI
        self._init_ui()
        self._init_connections()
        self._apply_stylesheet()
        
        # 初始化日志
        self._setup_logging()
    
    def _init_ui(self) -> None:
        """初始化 UI 组件"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 创建菜单栏
        self._create_menu_bar()
        
        # 创建主内容区
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        content_splitter.setStretchFactor(0, 3)  # 左侧
        content_splitter.setStretchFactor(1, 2)  # 中间
        content_splitter.setStretchFactor(2, 3)  # 右侧
        
        # 左侧: 音频输入面板
        self.audio_input_panel = AudioInputPanel()
        content_splitter.addWidget(self.audio_input_panel)
        
        # 中间: 模型配置面板
        self.model_config_panel = ModelConfigPanel()
        content_splitter.addWidget(self.model_config_panel)
        
        # 右侧: 输出面板
        self.output_panel = OutputPanel()
        content_splitter.addWidget(self.output_panel)
        
        main_layout.addWidget(content_splitter, stretch=1)
        
        # 底部: 状态栏
        self.status_bar = StatusBar()
        self.setStatusBar(self.status_bar)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
    
    def _create_menu_bar(self) -> None:
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")
        
        open_action = QAction("打开音频...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open_file)
        file_menu.addAction(open_action)
        
        save_action = QAction("导出音频...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._on_save_file)
        save_action.setEnabled(False)
        self._save_action = save_action
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 编辑菜单
        edit_menu = menubar.addMenu("编辑(&E)")
        
        preferences_action = QAction("首选项...", self)
        preferences_action.triggered.connect(self._on_preferences)
        edit_menu.addAction(preferences_action)
        
        # 工具菜单
        tools_menu = menubar.addMenu("工具(&T)")
        
        batch_action = QAction("批量转换...", self)
        batch_action.triggered.connect(self._on_batch_convert)
        tools_menu.addAction(batch_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")
        
        about_action = QAction("关于 SOMA...", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)
        
        documentation_action = QAction("使用文档", self)
        documentation_action.triggered.connect(self._on_documentation)
        help_menu.addAction(documentation_action)
    
    def _init_connections(self) -> None:
        """初始化信号连接"""
        # 音频输入面板
        self.audio_input_panel.file_selected.connect(self._on_input_file_selected)
        self.audio_input_panel.play_requested.connect(self._on_play_audio)
        
        # 模型配置面板
        self.model_config_panel.engine_changed.connect(self._on_engine_changed)
        self.model_config_panel.conversion_requested.connect(self._on_start_conversion)
        
        # 输出面板
        self.output_panel.export_requested.connect(self._on_export_audio)
        self.output_panel.play_requested.connect(self._on_play_output)
    
    def _apply_stylesheet(self) -> None:
        """应用样式表"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                color: #e0e0e0;
                font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
                font-size: 10pt;
            }
            QGroupBox {
                border: 1px solid #3c3c3c;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: #4fc3f7;
            }
            QPushButton {
                background-color: #0d47a1;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                color: white;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QPushButton:pressed {
                background-color: #0a3d91;
            }
            QPushButton:disabled {
                background-color: #424242;
                color: #757575;
            }
            QPushButton#primary {
                background-color: #1976d2;
            }
            QPushButton#primary:hover {
                background-color: #1e88e5;
            }
            QPushButton#danger {
                background-color: #c62828;
            }
            QPushButton#danger:hover {
                background-color: #d32f2f;
            }
            QComboBox {
                background-color: #2d2d2d;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 5px 10px;
                min-width: 120px;
            }
            QComboBox:hover {
                border-color: #4fc3f7;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #757575;
            }
            QSlider::groove:horizontal {
                border: 1px solid #3c3c3c;
                height: 6px;
                background: #2d2d2d;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4fc3f7;
                border: 1px solid #4fc3f7;
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: #4fc3f7;
                border-radius: 3px;
            }
            QProgressBar {
                border: 1px solid #3c3c3c;
                border-radius: 5px;
                text-align: center;
                background-color: #2d2d2d;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #4fc3f7;
                border-radius: 4px;
            }
            QLabel {
                color: #b0b0b0;
            }
            QListWidget {
                background-color: #252525;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 5px;
            }
            QListWidget::item:selected {
                background-color: #0d47a1;
            }
            QMenuBar {
                background-color: #252525;
            }
            QMenuBar::item {
                padding: 5px 10px;
            }
            QMenuBar::item:selected {
                background-color: #0d47a1;
            }
            QMenu {
                background-color: #252525;
                border: 1px solid #3c3c3c;
            }
            QMenu::item {
                padding: 5px 30px;
            }
            QMenu::item:selected {
                background-color: #0d47a1;
            }
            QStatusBar {
                background-color: #252525;
                color: #b0b0b0;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #2d2d2d;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 3px 5px;
            }
            QSpinBox:hover, QDoubleSpinBox:hover {
                border-color: #4fc3f7;
            }
        """)
    
    def _setup_logging(self) -> None:
        """设置日志记录"""
        try:
            from src.utils.logger import setup_logging
            setup_logging("INFO")
        except ImportError:
            pass  # 日志模块不可用时静默跳过
    
    # ==================== 事件处理 ====================
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 取消正在进行的转换
        if self._conversion_worker and self._conversion_worker.isRunning():
            self._conversion_worker.stop()
            self._conversion_worker.wait()
        
        event.accept()
    
    # ==================== 菜单动作 ====================
    
    def _on_open_file(self) -> None:
        """打开文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "打开音频文件",
            "",
            "音频文件 (*.wav *.mp3 *.flac *.ogg *.m4a);;所有文件 (*.*)"
        )
        
        if file_path:
            self.audio_input_panel.set_file(file_path)
    
    def _on_save_file(self) -> None:
        """保存文件"""
        if not self._output_file or not os.path.exists(self._output_file):
            return
        
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出音频文件",
            os.path.splitext(self._input_file)[0] + "_converted.wav" if self._input_file else "output.wav",
            "WAV 文件 (*.wav);;MP3 文件 (*.mp3);;所有文件 (*.*)"
        )
        
        if save_path:
            try:
                import shutil
                shutil.copy(self._output_file, save_path)
                self.status_bar.showMessage(f"已导出到: {save_path}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"导出音频时出错:\n{str(e)}")
    
    def _on_preferences(self) -> None:
        """打开首选项"""
        QMessageBox.information(self, "首选项", "首选项设置功能开发中...")
    
    def _on_batch_convert(self) -> None:
        """批量转换"""
        QMessageBox.information(self, "批量转换", "批量转换功能开发中...")
    
    def _on_about(self) -> None:
        """关于对话框"""
        QMessageBox.about(
            self,
            "关于 SOMA",
            "<h3>SOMA 智能音频工作站</h3>"
            "<p>版本: 0.1.0</p>"
            "<p>基于 AI 的音频处理工具，支持声音转换、音频分离、音效处理等功能。</p>"
            "<p>© 2024 SOMA Team</p>"
        )
    
    def _on_documentation(self) -> None:
        """打开文档"""
        QMessageBox.information(self, "使用文档", "使用文档功能开发中...")
    
    # ==================== 信号处理 ====================
    
    def _on_input_file_selected(self, file_path: str) -> None:
        """输入文件选择回调"""
        self._input_file = file_path
        self.status_bar.showMessage(f"已加载: {file_path}")
    
    def _on_engine_changed(self, engine: str) -> None:
        """引擎切换回调"""
        self._current_engine = engine
        self.status_bar.showMessage(f"当前引擎: {engine.upper()}")
    
    def _on_play_audio(self, file_path: str) -> None:
        """播放音频"""
        # TODO: 实现音频播放功能
        self.status_bar.showMessage(f"播放: {file_path}")
    
    def _on_start_conversion(self, params: dict) -> None:
        """开始转换"""
        if not self._input_file:
            QMessageBox.warning(self, "未选择文件", "请先选择要转换的音频文件")
            return
        
        # 创建工作线程
        self._conversion_worker = ConversionWorker(
            input_file=self._input_file,
            engine=self._current_engine,
            model_path=params.get("model_path", ""),
            f0_up_key=params.get("f0_up_key", 0),
            index_ratio=params.get("index_ratio", 0.5),
            filter_radius=params.get("filter_radius", 3),
            rms_mix=params.get("rms_mix", 0.5),
            pitch_algo=params.get("pitch_algo", "rmvpe"),
            protect=params.get("protect", 0.33),
        )
        
        # 连接信号
        self._conversion_worker.progress.connect(self._on_conversion_progress)
        self._conversion_worker.finished.connect(self._on_conversion_finished)
        self._conversion_worker.error.connect(self._on_conversion_error)
        
        # 更新 UI
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._save_action.setEnabled(False)
        self.model_config_panel.set_converting(True)
        
        # 启动线程
        self._conversion_worker.start()
        self.conversion_started.emit()
        self.status_bar.showMessage("转换中...")
    
    @pyqtSlot(int)
    def _on_conversion_progress(self, value: int) -> None:
        """转换进度更新"""
        self.progress_bar.setValue(value)
    
    @pyqtSlot(str)
    def _on_conversion_finished(self, output_file: str) -> None:
        """转换完成"""
        self._output_file = output_file
        self.progress_bar.setVisible(False)
        self._save_action.setEnabled(True)
        self.model_config_panel.set_converting(False)
        self.output_panel.set_output_file(output_file)
        self.status_bar.showMessage("转换完成")
        self.conversion_finished.emit(output_file)
    
    @pyqtSlot(str)
    def _on_conversion_error(self, error: str) -> None:
        """转换错误"""
        self.progress_bar.setVisible(False)
        self._save_action.setEnabled(False)
        self.model_config_panel.set_converting(False)
        self.status_bar.showMessage("转换失败")
        QMessageBox.critical(self, "转换错误", f"转换过程中出错:\n{error}")
        self.conversion_error.emit(error)
    
    def _on_export_audio(self, file_path: str) -> None:
        """导出音频"""
        self._output_file = file_path
        self._on_save_file()
    
    def _on_play_output(self, file_path: str) -> None:
        """播放输出音频"""
        self._on_play_audio(file_path)


def run_gui() -> None:
    """运行 GUI 应用"""
    app = QApplication(sys.argv)
    app.setApplicationName("SOMA")
    app.setOrganizationName("SOMA Team")
    
    # 设置 Qt 样式
    app.setStyle("Fusion")
    
    # 创建并显示主窗口
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    run_gui()
