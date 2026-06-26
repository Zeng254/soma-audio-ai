# RVC Training Module

SOMA RVC (Retrieval-Based Voice Conversion) dry voice model training pipeline.

## Overview

This module provides a complete training pipeline for RVC voice conversion models:

1. **Preprocessing** - Load, trim, normalize, and segment audio files
2. **Training** - Train generator + discriminator with AMP and gradient clipping
3. **Export** - Export trained models compatible with `RVCConverter`

## Quick Start

### 1. Preprocess Audio

```bash
# Using CLI
python -m src.training.cli preprocess raw_audio/ processed_data/ --sample-rate 40000

# Using Python API
from src.training import AudioPreprocessor, DataConfig

config = DataConfig(sample_rate=40000, segment_duration=3.0)
preprocessor = AudioPreprocessor(config)
stats = preprocessor.process_directory("raw_audio/", "processed_data/")
```

### 2. Train Model

```bash
# Using CLI
python -m src.training.cli train processed_data/ --epochs 100 --batch-size 8

# Using Python API
from src.training import TrainingConfig, RVCTrainer, RVCDataset, create_dataloader

config = TrainingConfig()
config.train.num_epochs = 100
config.train.batch_size = 8

dataset = RVCDataset("processed_data/", config.data)
dataloader = create_dataloader(dataset, batch_size=8)

trainer = RVCTrainer(config)
trainer.build_models()
trainer.build_optimizers()
trainer.train(dataloader)
```

### 3. Export for Inference

```bash
# Using CLI
python -m src.training.cli export --checkpoint output/checkpoints/checkpoint_best.pt --output output/model.pt

# Using Python API
trainer.export_for_inference("output/model.pt")

# Load with RVCConverter
from src.voice_converters.rvc_converter import RVCConverter
converter = RVCConverter()
converter.load_model("output/model.pt")
```

## Configuration

### Config File (JSON)

```json
{
  "data": {
    "sample_rate": 40000,
    "hop_length": 512,
    "n_mels": 128,
    "segment_duration": 3.0,
    "overlap": 0.1,
    "normalize_mode": "peak"
  },
  "model": {
    "in_channels": 256,
    "hidden_channels": 256,
    "use_flow": false
  },
  "optimizer": {
    "optimizer": "adamw",
    "lr": 2e-4,
    "scheduler": "cosine"
  },
  "train": {
    "batch_size": 8,
    "num_epochs": 1000,
    "use_amp": true,
    "device": "auto"
  },
  "f0": {
    "method": "dio",
    "f0_min": 50.0,
    "f0_max": 1100.0
  }
}
```

### Python API

```python
from src.training import TrainingConfig

config = TrainingConfig()
config.data.sample_rate = 40000
config.model.hidden_channels = 256
config.optimizer.lr = 2e-4
config.train.batch_size = 8

# Save / Load
config.save_json("my_config.json")
config = TrainingConfig.load_json("my_config.json")
```

## CLI Reference

### preprocess

```
python -m src.training.cli preprocess <input_dir> <output_dir> [options]

Options:
  --config PATH           Training config JSON
  --sample-rate INT       Target sample rate (default: 40000)
  --segment-duration FLOAT Segment duration in seconds (default: 3.0)
  --overlap FLOAT         Overlap ratio (default: 0.1)
  --silence-threshold FLOAT Silence threshold dB (default: -40.0)
  --normalize-mode {peak,lufs} Normalization mode (default: peak)
  --no-recursive          Don't search subdirectories
```

### train

```
python -m src.training.cli train <data_dir> [options]

Options:
  --config PATH           Training config JSON
  --batch-size INT        Batch size
  --epochs INT            Number of epochs
  --device {cpu,cuda,mps} Compute device
  --output-dir PATH       Output directory
  --train-ratio FLOAT     Train/val split ratio (default: 0.9)
  --resume PATH           Resume from checkpoint
  --no-tensorboard        Disable TensorBoard
```

### export

```
python -m src.training.cli export --checkpoint <path> [options]

Options:
  --checkpoint PATH       Training checkpoint (required)
  --output PATH           Export output path
  --config PATH           Training config JSON
  --device STR            Device for export (default: cpu)
```

## Architecture

### Generator
- Shared encoder (Projection + ResBlocks)
- F0 condition modulation
- Upsample decoder
- Final convolution output

### Discriminator
- Multi-Period Discriminator (MPD)
- Multiple period groups for temporal pattern capture

### Loss Functions
- **Mel Spectrogram Loss** (weight: 45.0)
- **L1 Loss** (weight: 1.0)
- **Adversarial Loss** (weight: 1.0)
- **Feature Matching Loss** (weight: 2.0)

## Output Structure

```
output/rvc_training/
├── training_config.json
├── logs/                  # TensorBoard logs
│   └── events.out.tfevents.*
└── checkpoints/
    ├── checkpoint_latest.pt
    ├── checkpoint_best.pt
    └── checkpoint_epoch_0100.pt
```

## Model Compatibility

Exported models are compatible with `RVCConverter`:

```python
from src.voice_converters.rvc_converter import RVCConverter

converter = RVCConverter(device="cuda")
converter.load_model("output/model.pt")
result = converter.convert("input.wav", "output.wav")
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- NumPy >= 1.24
- librosa (optional, for advanced preprocessing)
- soundfile (optional, fallback audio loading)
- tensorboard (optional, for training visualization)
