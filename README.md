# SOMA - AI 驱动的音频处理工作站

<p align="center">
  <img src="assets/logo.png" alt="SOMA Logo" width="200" style="background: transparent;"/>
</p>

<p align="center">
  <strong>SOMA</strong> — 智能音频处理工作站
</p>

---

## 📖 项目简介

**SOMA** (Smart Optical Music & Audio) 是一个基于 AI 的音频处理工作站，集成了业界领先的音频分离、音效处理、格式转换等功能。通过模块化的设计，SOMA 能够满足从专业音乐制作到日常音频编辑的各种需求。

### 核心特性

- 🎵 **智能声音分离** — 基于深度学习的人声/伴奏分离、鼓点/贝斯/其他乐器提取
- 🎙️ **声音转换** — 双引擎架构支持 RVC v2 / So-VITS-SVC 4.1
- 🎛️ **专业音效处理** — 多频段均衡器、混响效果、音调变换
- 🔄 **格式转换** — 支持多种音频格式的高质量转换
- ⚡ **流水线处理** — 链式调用多个处理节点，批量自动化处理
- 🧠 **AI 增强** — 内置 Agent 能力，支持自然语言控制音频处理流程

---

## 🏗️ 项目结构

```
src/
├── separators/        # 音频分离器模块
│   ├── base.py              # 分离器基类
│   ├── demucs_separator.py  # Demucs 分离器实现
│   └── msst_separator.py    # MSST 分离器实现
│
├── voice_converters/  # 声音转换模块 (双引擎架构)
│   ├── base.py              # 转换器基类和通用接口
│   ├── rvc_converter.py     # RVC v2 引擎实现
│   ├── sovits_converter.py  # So-VITS-SVC 4.1 引擎实现
│   └── factory.py           # 引擎工厂和自动识别
│
├── effects/         # 音效处理模块
│   ├── base.py              # 效果器基类
│   ├── eq.py                # 多频段均衡器
│   ├── reverb.py            # 混响效果器
│   └── pitch.py             # 音调变换器
│
├── converters/       # 格式转换模块
│   └── converter.py         # 音频格式转换器
│
├── pipeline/         # 处理流水线模块
│   └── pipeline.py          # 链式处理流水线
│
└── utils/           # 工具模块
    ├── audio_io.py          # 音频读写工具
    └── validator.py         # 参数校验工具
```

---

## 🚀 快速开始

### 安装依赖

```bash
uv sync
```

### 基本使用

#### 1. 声音分离

```python
from src.separators import DemucsSeparator
from src.utils.audio_io import AudioLoader, AudioSaver

# 初始化分离器
separator = DemucsSeparator(model_name="htdemucs")

# 加载音频
loader = AudioLoader()
audio, sr = loader.load("input.mp3")

# 执行分离
result = separator.separate_array(audio, sr)

# 保存分离结果
saver = AudioSaver()
if result.vocals is not None:
    saver.save(result.vocals, "vocals.wav", sr)
if result.accompaniment is not None:
    saver.save(result.accompaniment, "accompaniment.wav", sr)
```

#### 2. 音效处理

```python
from src.effects import Equalizer, Reverb, PitchShifter

# 均衡器
eq = Equalizer(sample_rate=44100)
eq.set_preset("pop")
result = eq.process(audio, sr)

# 混响
reverb = Reverb(sample_rate=44100, room_size=0.6, wet_level=0.3)
result = reverb.process(audio, sr)

# 音调变换
pitch = PitchShifter(sample_rate=44100)
result = pitch.process(audio, sr, semitones=2)  # 升高两个半音
```

#### 3. 声音转换

```python
from src.voice_converters import create_converter, ConversionParams

# 自动识别并创建转换器
converter = create_converter(
    model_path="path/to/model.pth",
    device="cuda"
)

# 设置转换参数
params = ConversionParams(
    pitch_shift=0,       # 半音调整 (-24 to +24)
    pitch_algo="rmvpe",  # 音高算法
    vpm=0.5,             # 音色匹配 (0.0-1.0)
    rms_mix=0.5,         # 响度混合
)

# 执行转换
result = converter.convert(audio, sample_rate, params)

# 使用上下文管理器
with create_converter("path/to/model.pth") as vc:
    result = vc.convert(audio, sr)
```

#### 4. 流水线处理

```python
from src.pipeline import PipelineBuilder
from src.separators import DemucsSeparator
from src.effects import Equalizer, Reverb

# 构建处理流水线
pipeline = (
    PipelineBuilder(name="music_processing")
    .with_separator("split", DemucsSeparator())
    .with_effect("eq", Equalizer(), preset="vocal_boost")
    .with_effect("reverb", Reverb(), room_size=0.5)
    .build()
)

# 执行流水线
result = pipeline.execute(audio, sr)
```

---

## 📦 核心模块

### Separators（分离器模块）

| 模块 | 功能 | 模型 |
|------|------|------|
| `DemucsSeparator` | 人声/鼓/贝斯/其他分离 | htdemucs |
| `MSSTSeparator` | 高保真人声/伴奏分离 | MSST |

### Voice Converters（声音转换模块）

| 模块 | 功能 | 特性 |
|------|------|------|
| `RVCConverter` | RVC v2 声音转换 | 检索增强、索引加速 |
| `SoVITSConverter` | So-VITS-SVC 4.1 转换 | 扩散模式、音色保护 |
| `ConverterFactory` | 引擎工厂 | 自动识别、统一接口 |

### Effects（效果器模块）

| 模块 | 功能 | 参数 |
|------|------|------|
| `Equalizer` | 多频段均衡器 | 10段预设/自定义 |
| `Reverb` | 混响效果 | room/hall/plate/cathedral |
| `PitchShifter` | 音调变换 | ±24半音 |

### Converters（转换器模块）

| 模块 | 功能 |
|------|------|
| `AudioConverter` | 格式转换、采样率转换、批量处理 |

### Pipeline（流水线模块）

| 模块 | 功能 |
|------|------|
| `AudioPipeline` | 链式处理、节点管理 |
| `PipelineBuilder` | 流畅 API 构建 |

---

## 🔧 开发指南

### 运行流程

```bash
# 本地运行
bash scripts/local_run.sh -m flow

# 运行节点
bash scripts/local_run.sh -m node -n node_name

# 启动 HTTP 服务
bash scripts/http_run.sh -m http -p 5000
```

### 添加新的效果器

1. 继承 `BaseEffect` 基类
2. 实现 `process` 和 `get_effect_name` 方法
3. 在 `__init__.py` 中导出

```python
from src.effects.base import BaseEffect, EffectResult

class MyEffect(BaseEffect):
    def get_effect_name(self) -> str:
        return "MyEffect"
    
    def process(self, audio, sample_rate, **kwargs):
        # 实现处理逻辑
        return self._create_result(output_audio, sample_rate)
```

---

## 📋 环境要求

- Python >= 3.12
- FFmpeg (用于音频格式转换)
- PyTorch (用于 AI 模型推理)

### 可选依赖

```bash
# 音频处理
uv add librosa soundfile pydub

# AI 模型
uv add demucs torch torchaudio

# Web 服务
uv add fastapi uvicorn
```

---

## 📄 许可证

本项目基于 MIT 许可证开源。

---

<p align="center">
  <strong>SOMA</strong> — 让音频处理更智能
</p>
