# SOMA 打包与部署指南

## 目录

- [环境准备](#环境准备)
- [依赖安装](#依赖安装)
- [本地测试](#本地测试)
- [打包发布](#打包发布)
- [跨平台打包](#跨平台打包)
- [常见问题](#常见问题)

---

## 环境准备

### 系统要求

| 组件 | 要求 |
|------|------|
| Python | 3.10+ (推荐 3.12) |
| 内存 | 最少 8GB (推荐 16GB+) |
| 显存 | 可选，GPU 加速需要 6GB+ |
| 磁盘 | 最少 10GB 可用空间 |

### 安装 Python

```bash
# Windows
# 下载 python-3.12.x.exe from https://www.python.org/downloads/
# 安装时勾选 "Add Python to PATH"

# Linux (Ubuntu/Debian)
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev

# macOS
brew install python@3.12
```

### 创建虚拟环境

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

# 升级 pip
pip install --upgrade pip
```

---

## 依赖安装

### 安装所有依赖

```bash
# 安装核心依赖
pip install -e ".[core]"

# 安装开发依赖
pip install -e ".[dev]"

# 安装 GUI 依赖 (可选)
pip install -e ".[gui]"

# 安装所有依赖
pip install -e ".[all]"
```

### 单独安装开发依赖

```bash
pip install pytest pytest-cov pytest-asyncio
pip install pyinstaller
```

---

## 本地测试

### 运行所有测试

```bash
# 运行所有测试
pytest

# 带详细输出
pytest -v

# 带覆盖率
pytest --cov=src --cov-report=html

# 只运行特定模块
pytest tests/test_config/
pytest tests/test_security/
```

### 测试特定功能

```bash
# 运行配置相关测试
pytest tests/test_config/test_config.py -v

# 运行安全相关测试
pytest tests/test_security/test_security.py -v

# 运行效果器测试
pytest tests/test_effects/test_effects.py -v

# 跳过慢速测试
pytest -m "not slow"

# 只运行需要 GPU 的测试
pytest -m "requires_gpu"
```

---

## 打包发布

### 方式一：PyInstaller 打包 (推荐)

#### 安装 PyInstaller

```bash
pip install pyinstaller
```

#### 打包为可执行文件

```bash
# 进入项目目录
cd soma-audio-ai

# 打包 (目录模式)
pyinstaller soma.spec

# 打包 (单文件模式)
# 编辑 soma.spec 将 ONE_FILE = True
pyinstaller soma.spec --onefile
```

#### 打包输出

```
dist/
├── SOMA/                    # 目录模式
│   ├── SOMA.exe            # Windows
│   ├── SOMA                # Linux/macOS
│   ├── QtCore.so
│   ├── PyQt6/
│   └── ...
└── SOMA                    # 单文件模式 (Windows)
```

#### 运行打包后的程序

```bash
# Windows
dist\SOMA\SOMA.exe

# Linux
chmod +x dist/SOMA/SOMA
./dist/SOMA/SOMA
```

### 方式二：直接安装

```bash
# 构建分发包
python -m build

# 安装
pip install dist/soma_audio_ai-*.whl
```

### 方式三：Docker 部署

#### 构建镜像

```bash
docker build -t soma-audio-ai:latest .
```

#### 运行容器

```bash
# 运行容器
docker run -it --gpus all soma-audio-ai:latest

# 挂载本地目录
docker run -it --gpus all \
    -v /path/to/audio:/app/audio \
    -v /path/to/models:/app/models \
    soma-audio-ai:latest
```

---

## 跨平台打包

### Windows 打包 (在 Windows 上运行)

```bash
# 安装依赖
pip install -e ".[all]"

# 打包
pyinstaller soma.spec
```

### Linux 打包

```bash
# 安装依赖
pip install -e ".[all]"

# 打包
pyinstaller soma.spec
```

### macOS 打包

```bash
# 安装依赖
pip install -e ".[all]"

# 打包
pyinstaller soma.spec
```

### 使用 GitHub Actions 跨平台打包

创建 `.github/workflows/release.yml`:

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    strategy:
      matrix:
        platform: [windows-latest, ubuntu-latest, macos-latest]

    runs-on: ${{ matrix.platform }}

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install uv
        uses: astral-sh/setup-uv@v3

      - name: Install dependencies
        run: uv sync --all-extras

      - name: Build executable
        run: |
          pip install pyinstaller
          pyinstaller soma.spec

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: soma-${{ matrix.platform }}
          path: dist/
```

---

## 常见问题

### Q1: 打包后程序启动失败

**症状**: 双击程序无反应或报错

**解决方案**:
1. 从命令行运行，查看错误信息
2. 检查是否缺少 DLL
3. 确认 PyQt6 已正确打包

### Q2: 缺少模型文件

**症状**: `Model not found` 错误

**解决方案**:
1. 将模型文件放在 `models/` 目录
2. 使用绝对路径指定模型位置

### Q3: GPU 不可用

**症状**: `CUDA not available`

**解决方案**:
1. 安装 NVIDIA 驱动
2. 安装 CUDA Toolkit
3. 确认 PyTorch 支持 CUDA

### Q4: 打包体积过大

**解决方案**:
1. 使用虚拟环境隔离依赖
2. 排除不需要的模块 (修改 spec 文件)
3. 使用 UPX 压缩

### Q5: macOS 无法运行未签名应用

**解决方案**:
```bash
# 允许运行未签名应用
sudo spctl --master-disable

# 或者右键选择"打开"
```

---

## 目录结构参考

打包后的目录结构:

```
dist/SOMA/
├── SOMA                      # 主程序
├── QtCore.so                # Qt 核心库
├── PyQt6/                   # PyQt6 库
├── matplotlib/              # 可选
├── torch/                   # PyTorch
├── numpy/                   # NumPy
├── config/                  # 配置文件
├── models/                  # 模型目录 (用户放置)
│   ├── rvc/
│   └── sovits/
├── outputs/                 # 输出目录 (用户放置)
└── resources/               # 资源文件
```

---

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| 0.1.0 | 2024-xx-xx | 初始版本 |

---

## 许可证

本项目使用 MIT 许可证。详见 [LICENSE](LICENSE)。
