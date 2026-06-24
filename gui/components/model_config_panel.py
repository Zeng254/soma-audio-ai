"""
模型配置面板

功能:
- 选择引擎类型 (RVC / SoVITS)
- 选择模型文件
- 调节转换参数
"""

import os
from typing import Optional, Dict

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFileDialog, QGroupBox,
    QComboBox, QSlider, QCheckBox, QSpinBox,
    QDoubleSpinBox, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize


class ModelConfigPanel(QWidget):
    """
    模型配置面板
    
    Signals:
        engine_changed(str): 引擎类型改变
        conversion_requested(dict): 请求开始转换
    """
    
    engine_changed = pyqtSignal(str)
    conversion_requested = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._current_engine = "rvc"
        self._is_converting = False
        
        self._init_ui()
    
    def _init_ui(self) -> None:
        """初始化 UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 引擎选择
        engine_group = QGroupBox("引擎选择")
        engine_layout = QVBoxLayout()
        
        engine_select_layout = QHBoxLayout()
        engine_select_layout.addWidget(QLabel("引擎:"))
        
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["RVC v2", "SoVITS-SVC 4.1"])
        self.engine_combo.setCurrentText("RVC v2")
        self.engine_combo.currentTextChanged.connect(self._on_engine_changed)
        engine_select_layout.addWidget(self.engine_combo, stretch=1)
        
        engine_layout.addLayout(engine_select_layout)
        engine_group.setLayout(engine_layout)
        layout.addWidget(engine_group)
        
        # 模型选择
        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout()
        
        # RVC 模型
        self.rvc_model_layout = QVBoxLayout()
        
        rvc_row1 = QHBoxLayout()
        rvc_row1.addWidget(QLabel("模型文件:"))
        self.rvc_model_btn = QPushButton("选择 .pth 文件")
        self.rvc_model_btn.clicked.connect(lambda: self._select_model("rvc"))
        rvc_row1.addWidget(self.rvc_model_btn)
        self.rvc_model_layout.addLayout(rvc_row1)
        
        self.rvc_model_label = QLabel("未选择")
        self.rvc_model_label.setStyleSheet("color: #757575;")
        self.rvc_model_layout.addWidget(self.rvc_model_label)
        
        rvc_index_row = QHBoxLayout()
        rvc_index_row.addWidget(QLabel("索引文件:"))
        self.rvc_index_btn = QPushButton("选择 .index 文件")
        self.rvc_index_btn.clicked.connect(lambda: self._select_index("rvc"))
        rvc_index_row.addWidget(self.rvc_index_btn)
        self.rvc_index_layout = QHBoxLayout()
        self.rvc_index_label = QLabel("可选")
        self.rvc_index_label.setStyleSheet("color: #757575;")
        self.rvc_model_layout.addWidget(self.rvc_index_label)
        rvc_index_row.addWidget(self.rvc_index_label)
        self.rvc_model_layout.addLayout(rvc_index_row)
        
        model_layout.addLayout(self.rvc_model_layout)
        
        # SoVITS 模型
        self.sovits_model_layout = QVBoxLayout()
        self.sovits_model_layout.setEnabled(False)
        
        sovits_row1 = QHBoxLayout()
        sovits_row1.addWidget(QLabel("模型文件:"))
        self.sovits_model_btn = QPushButton("选择 .pth 文件")
        self.sovits_model_btn.clicked.connect(lambda: self._select_model("sovits"))
        sovits_row1.addWidget(self.sovits_model_btn)
        self.sovits_model_layout.addLayout(sovits_row1)
        
        self.sovits_model_label = QLabel("未选择")
        self.sovits_model_label.setStyleSheet("color: #757575;")
        self.sovits_model_layout.addWidget(self.sovits_model_label)
        
        sovits_config_row = QHBoxLayout()
        sovits_config_row.addWidget(QLabel("配置文件:"))
        self.sovits_config_btn = QPushButton("选择 .json 文件")
        self.sovits_config_btn.clicked.connect(lambda: self._select_config())
        sovits_config_row.addWidget(self.sovits_config_btn)
        self.sovits_model_layout.addLayout(sovits_config_row)
        
        self.sovits_config_label = QLabel("未选择")
        self.sovits_config_label.setStyleSheet("color: #757575;")
        self.sovits_model_layout.addWidget(self.sovits_config_label)
        
        model_layout.addLayout(self.sovits_model_layout)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 转换参数
        params_group = QGroupBox("转换参数")
        params_layout = QGridLayout()
        
        # 音高调整
        params_layout.addWidget(QLabel("音高升降:"), 0, 0)
        self.f0_spin = QDoubleSpinBox()
        self.f0_spin.setRange(-24, 24)
        self.f0_spin.setValue(0)
        self.f0_spin.setSuffix(" 半音")
        self.f0_spin.setDecimals(1)
        params_layout.addWidget(self.f0_spin, 0, 1)
        
        # F0 算法
        params_layout.addWidget(QLabel("F0 算法:"), 1, 0)
        self.f0_algo_combo = QComboBox()
        self.f0_algo_combo.addItems(["rmvpe", "crepe", "dio", "harvest", "pm"])
        self.f0_algo_combo.setCurrentText("rmvpe")
        params_layout.addWidget(self.f0_algo_combo, 1, 1)
        
        # 索引比率
        params_layout.addWidget(QLabel("索引比率:"), 2, 0)
        self.index_ratio_slider = QSlider(Qt.Orientation.Horizontal)
        self.index_ratio_slider.setRange(0, 100)
        self.index_ratio_slider.setValue(50)
        params_layout.addWidget(self.index_ratio_slider, 2, 1)
        self.index_ratio_label = QLabel("0.50")
        params_layout.addWidget(self.index_ratio_label, 2, 2)
        self.index_ratio_slider.valueChanged.connect(
            lambda v: self.index_ratio_label.setText(f"{v/100:.2f}")
        )
        
        # RMS 混合
        params_layout.addWidget(QLabel("响度混合:"), 3, 0)
        self.rms_mix_slider = QSlider(Qt.Orientation.Horizontal)
        self.rms_mix_slider.setRange(0, 100)
        self.rms_mix_slider.setValue(50)
        params_layout.addWidget(self.rms_mix_slider, 3, 1)
        self.rms_mix_label = QLabel("0.50")
        params_layout.addWidget(self.rms_mix_label, 3, 2)
        self.rms_mix_slider.valueChanged.connect(
            lambda v: self.rms_mix_label.setText(f"{v/100:.2f}")
        )
        
        # 保护音
        params_layout.addWidget(QLabel("保护音:"), 4, 0)
        self.protect_slider = QSlider(Qt.Orientation.Horizontal)
        self.protect_slider.setRange(0, 100)
        self.protect_slider.setValue(33)
        params_layout.addWidget(self.protect_slider, 4, 1)
        self.protect_label = QLabel("0.33")
        params_layout.addWidget(self.protect_label, 4, 2)
        self.protect_slider.valueChanged.connect(
            lambda v: self.protect_label.setText(f"{v/100:.2f}")
        )
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # 转换按钮
        self.convert_btn = QPushButton("开始转换")
        self.convert_btn.setObjectName("primary")
        self.convert_btn.clicked.connect(self._on_convert)
        layout.addWidget(self.convert_btn)
        
        layout.addStretch()
        
        # 保存组件引用
        self._params_group = params_group
        self._model_group = model_group
    
    def _on_engine_changed(self, text: str) -> None:
        """引擎切换"""
        engine = "rvc" if text.startswith("RVC") else "sovits"
        self._current_engine = engine
        
        # 切换模型选择区域
        self.rvc_model_layout.setEnabled(engine == "rvc")
        self.sovits_model_layout.setEnabled(engine == "sovits")
        
        self.engine_changed.emit(engine)
    
    def _select_model(self, engine: str) -> None:
        """选择模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"选择 {engine.upper()} 模型文件",
            "",
            "模型文件 (*.pth);;所有文件 (*.*)"
        )
        
        if file_path:
            if engine == "rvc":
                self.rvc_model_label.setText(os.path.basename(file_path))
                self.rvc_model_label.setStyleSheet("color: #4fc3f7;")
                self._rvc_model_path = file_path
            else:
                self.sovits_model_label.setText(os.path.basename(file_path))
                self.sovits_model_label.setStyleSheet("color: #4fc3f7;")
                self._sovits_model_path = file_path
    
    def _select_index(self, engine: str) -> None:
        """选择索引文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择索引文件",
            "",
            "索引文件 (*.index);;所有文件 (*.*)"
        )
        
        if file_path:
            self.rvc_index_label.setText(os.path.basename(file_path))
            self.rvc_index_label.setStyleSheet("color: #4fc3f7;")
            self._rvc_index_path = file_path
    
    def _select_config(self) -> None:
        """选择配置文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 SoVITS 配置文件",
            "",
            "配置文件 (*.json);;所有文件 (*.*)"
        )
        
        if file_path:
            self.sovits_config_label.setText(os.path.basename(file_path))
            self.sovits_config_label.setStyleSheet("color: #4fc3f7;")
            self._sovits_config_path = file_path
    
    def _on_convert(self) -> None:
        """开始转换"""
        # 获取模型路径
        if self._current_engine == "rvc":
            model_path = getattr(self, '_rvc_model_path', '')
            if not model_path:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "未选择模型", "请先选择 RVC 模型文件")
                return
        else:
            model_path = getattr(self, '_sovits_model_path', '')
            if not model_path:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "未选择模型", "请先选择 SoVITS 模型文件")
                return
        
        # 构建参数
        params = {
            "model_path": model_path,
            "f0_up_key": self.f0_spin.value(),
            "index_ratio": self.index_ratio_slider.value() / 100,
            "f0_algo": self.f0_algo_combo.currentText(),
            "rms_mix": self.rms_mix_slider.value() / 100,
            "protect": self.protect_slider.value() / 100,
        }
        
        # 添加引擎特定参数
        if self._current_engine == "rvc":
            params["index_path"] = getattr(self, '_rvc_index_path', '')
        else:
            params["config_path"] = getattr(self, '_sovits_config_path', '')
        
        self.conversion_requested.emit(params)
    
    def set_converting(self, converting: bool) -> None:
        """设置转换状态"""
        self._is_converting = converting
        self.convert_btn.setEnabled(not converting)
        self.convert_btn.setText("转换中..." if converting else "开始转换")
    
    def get_current_engine(self) -> str:
        """获取当前引擎"""
        return self._current_engine
