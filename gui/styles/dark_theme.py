"""
SOMA GUI 样式表

定义深色主题样式
"""

# 主窗口样式
MAIN_WINDOW_STYLE = """
QMainWindow {
    background-color: #121212;
}

QWidget {
    background-color: #121212;
    color: #e0e0e0;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 13px;
}

QLabel {
    color: #e0e0e0;
}

QGroupBox {
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 8px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #4fc3f7;
}
"""

# 按钮样式
BUTTON_STYLE = """
QPushButton {
    background-color: #2d2d2d;
    border: 1px solid #4a4a4a;
    border-radius: 4px;
    padding: 8px 16px;
    color: #e0e0e0;
    min-height: 30px;
}

QPushButton:hover {
    background-color: #3d3d3d;
    border-color: #4fc3f7;
}

QPushButton:pressed {
    background-color: #1d1d1d;
}

QPushButton:disabled {
    background-color: #1e1e1e;
    color: #5a5a5a;
    border-color: #3c3c3c;
}

QPushButton#primary {
    background-color: #4fc3f7;
    color: #121212;
    border-color: #4fc3f7;
    font-weight: bold;
}

QPushButton#primary:hover {
    background-color: #81d4fa;
}

QPushButton#primary:disabled {
    background-color: #3c3c3c;
    color: #757575;
    border-color: #3c3c3c;
}
"""

# 输入框样式
INPUT_STYLE = """
QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox {
    background-color: #1e1e1e;
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    padding: 5px 10px;
    color: #e0e0e0;
}

QLineEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #4fc3f7;
}

QLineEdit:disabled, QTextEdit:disabled {
    background-color: #1a1a1a;
    color: #5a5a5a;
}
"""

# 滑块样式
SLIDER_STYLE = """
QSlider::groove:horizontal {
    border: 1px solid #3c3c3c;
    height: 6px;
    background-color: #1e1e1e;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background-color: #4fc3f7;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background-color: #81d4fa;
}

QSlider::sub-page:horizontal {
    background-color: #4fc3f7;
    border-radius: 3px;
}

QSlider::add-page:horizontal {
    background-color: #3c3c3c;
    border-radius: 3px;
}
"""

# 进度条样式
PROGRESS_STYLE = """
QProgressBar {
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    text-align: center;
    background-color: #1e1e1e;
    height: 20px;
}

QProgressBar::chunk {
    background-color: #4fc3f7;
    border-radius: 3px;
}
"""

# 下拉框样式
COMBO_STYLE = """
QComboBox {
    background-color: #1e1e1e;
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    padding: 5px 10px;
    color: #e0e0e0;
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
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 5px solid #757575;
}

QComboBox QAbstractItemView {
    background-color: #1e1e1e;
    border: 1px solid #3c3c3c;
    selection-background-color: #4fc3f7;
    color: #e0e0e0;
}
"""

# 滚动条样式
SCROLLBAR_STYLE = """
QScrollBar:vertical {
    background-color: #1e1e1e;
    width: 12px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background-color: #3c3c3c;
    min-height: 20px;
    border-radius: 6px;
}

QScrollBar::handle:vertical:hover {
    background-color: #4a4a4a;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background-color: #1e1e1e;
    height: 12px;
    margin: 0;
}

QScrollBar::handle:horizontal {
    background-color: #3c3c3c;
    min-width: 20px;
    border-radius: 6px;
}
"""

# 状态栏样式
STATUSBAR_STYLE = """
QStatusBar {
    background-color: #1e1e1e;
    border-top: 1px solid #3c3c3c;
    color: #757575;
}

QStatusBar::item {
    border: none;
}
"""

# 组合所有样式
APP_STYLE = "\n".join([
    MAIN_WINDOW_STYLE,
    BUTTON_STYLE,
    INPUT_STYLE,
    SLIDER_STYLE,
    PROGRESS_STYLE,
    COMBO_STYLE,
    SCROLLBAR_STYLE,
    STATUSBAR_STYLE,
])
