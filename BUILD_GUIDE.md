# SOMA AI Workstation - Windows 打包指南

## 快速开始

### 一键打包（推荐）

```bash
# Windows
build_windows.bat

# Linux/macOS
chmod +x build.sh
./build.sh
```

打包完成后，在 `dist/` 目录下会生成：
- `SOMA/` - 可执行目录
- `SOMA-v0.1.0-win64.zip` - 分发包

### 用户使用方法

1. 解压 `SOMA-v0.1.0-win64.zip` 到任意目录
2. 双击 `SOMA.exe` 启动应用
3. 首次启动会自动创建 `~/.soma/` 目录存储模型和输出

---

## 打包流程详解

### 1. 环境准备

```bash
# 安装 Python 3.12+
# Windows: 从 https://www.python.org/downloads/ 下载安装
# 安装时勾选 "Add Python to PATH"

# 安装 uv 包管理器
pip install uv

# 安装项目依赖
uv sync --extra dev
```

### 2. 构建命令

```bash
# 完整构建（安装依赖 + 构建）
build_windows.bat

# 快速构建（跳过依赖安装）
build_windows.bat --quick

# 清理构建（删除旧文件后重新构建）
build_windows.bat --clean
```

### 3. 构建产物

```
dist/
├── SOMA/                    # 可执行目录
│   ├── SOMA.exe            # 主程序
│   ├── _internal/          # 运行时依赖
│   ├── models/             # 模型目录（用户需自行添加）
│   ├── output/             # 输出目录
│   └── README.txt          # 使用说明
└── SOMA-v0.1.0-win64.zip   # 分发包
```

---

## 目录结构说明

### 打包后的目录结构

```
SOMA/
├── SOMA.exe                # 主程序入口
├── _internal/              # PyInstaller 运行时文件
│   ├── base_library.zip    # Python 标准库
│   ├── torch/              # PyTorch 运行时
│   ├── numpy/              # NumPy
│   ├── scipy/              # SciPy
│   ├── librosa/            # Librosa
│   └── ...                 # 其他依赖
├── models/                 # 模型文件目录（空）
├── output/                 # 输出文件目录（空）
└── README.txt              # 使用说明
```

### 用户数据目录

应用运行时会在用户目录下创建：

```
~/.soma/                    # Windows: C:\Users\<用户名>\.soma\
├── models/                 # RVC/SoVITS 模型文件
├── output/                 # 转换输出文件
└── configs/                # 用户配置
```

---

## 自定义配置

### 修改应用图标

1. 准备 `.ico` 格式的图标文件
2. 放置到 `assets/icon.ico`
3. 修改 `soma.spec` 中的 `icon` 参数：

```python
exe = EXE(
    ...
    icon='assets/icon.ico',  # 取消注释并修改路径
    ...
)
```

### 修改版本信息

编辑 `version_info.txt`：

```
StringStruct('FileVersion', '0.2.0'),
StringStruct('ProductVersion', '0.2.0'),
```

同时更新 `pyproject.toml` 中的版本号。

### 排除更多模块以减小体积

编辑 `soma.spec` 中的 `excludes` 列表，添加不需要的模块：

```python
excludes = [
    'module_name',  # 添加要排除的模块
    ...
]
```

---

## 常见问题

### Q: 打包后体积很大怎么办？

A: PyTorch 是主要体积来源（约 200-400MB）。可以考虑：
- 使用 CPU-only 版本的 PyTorch（已在 pyproject.toml 中配置）
- 排除不需要的模块
- 使用 UPX 压缩（已启用）

### Q: 打包后运行报错 "ModuleNotFoundError"

A: 需要在 `soma.spec` 的 `hiddenimports` 中添加缺失的模块：

```python
hiddenimports = [
    'missing_module',  # 添加缺失的模块
    ...
]
```

### Q: 如何减小分发包体积？

A: 建议方案：
1. 使用 CPU-only PyTorch（已配置）
2. 排除不需要的依赖（见 `excludes`）
3. 使用 7-Zip 压缩代替 ZIP（压缩率更高）
4. 考虑将模型文件单独分发

### Q: 打包后无法找到模型文件

A: 模型文件不包含在分发包中，用户需要：
1. 将模型文件放到 `~/.soma/models/` 目录
2. 或在应用设置中指定模型目录

### Q: Windows Defender 报毒

A: PyInstaller 打包的 exe 可能被误报。解决方案：
1. 使用代码签名证书（需要购买）
2. 向 Microsoft 提交误报报告
3. 告知用户添加信任

---

## 技术细节

### PyInstaller 配置

- **入口文件**: `launcher.py`
- **打包模式**: 目录模式（`onedir`）
- **控制台**: 关闭（GUI 应用）
- **UPX 压缩**: 启用
- **隐藏导入**: 已配置所有动态导入的模块

### 路径处理

`launcher.py` 中的 `get_base_path()` 函数处理了两种运行模式：
- **开发模式**: 使用源码目录
- **打包模式**: 使用 `sys._MEIPASS`（PyInstaller 临时解压目录）

### 环境变量

打包后的应用会自动设置：
- `SOMA_HOME`: 用户数据目录（默认 `~/.soma`）

---

## 发布流程

1. 更新版本号
   ```bash
   # 修改 pyproject.toml 和 version_info.txt 中的版本号
   ```

2. 执行构建
   ```bash
   build_windows.bat --clean
   ```

3. 测试分发包
   ```bash
   # 在另一台机器上测试
   # 解压 ZIP 并运行 SOMA.exe
   ```

4. 发布
   - 上传 ZIP 到 GitHub Releases
   - 更新 CHANGELOG

---

## 参考文档

- [PyInstaller 官方文档](https://pyinstaller.org/en/stable/)
- [PyInstaller spec 文件说明](https://pyinstaller.org/en/stable/spec-files.html)
- [Python 打包最佳实践](https://packaging.python.org/)
