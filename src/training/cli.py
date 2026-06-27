"""
RVC Training CLI Module

Command-line interface for RVC model training pipeline:
- preprocess: Batch preprocess audio files
- train: Start model training
- export: Export trained model for inference (compatible with RVCConverter)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging for CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_preprocess(args):
    """Handle 'preprocess' subcommand."""
    from .config import DataConfig, TrainingConfig
    from .preprocess import AudioPreprocessor

    # Load or create config
    if args.config:
        config = TrainingConfig.load_json(args.config)
        data_config = config.data
    else:
        data_config = DataConfig(
            sample_rate=args.sample_rate,
            segment_duration=args.segment_duration,
            overlap=args.overlap,
            silence_threshold=args.silence_threshold,
            normalize_mode=args.normalize_mode,
        )

    preprocessor = AudioPreprocessor(data_config)

    logger.info("Starting preprocessing: %s -> %s", args.input_dir, args.output_dir)
    stats = preprocessor.process_directory(
        args.input_dir,
        args.output_dir,
        recursive=args.recursive,
    )

    logger.info("Preprocessing complete:")
    logger.info("  Total files: %d", stats["total_files"])
    logger.info("  Processed: %d", stats["processed_files"])
    logger.info("  Failed: %d", stats["failed_files"])
    logger.info("  Total segments: %d", stats["total_segments"])

    return stats


def cmd_train(args):
    """Handle 'train' subcommand."""
    from .config import TrainingConfig
    from .dataset import RVCDataset, create_dataloader, split_dataset
    from .trainer import RVCTrainer

    # Load config
    if args.config:
        config = TrainingConfig.load_json(args.config)
    else:
        config = TrainingConfig()

    # Override with CLI args
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.epochs:
        config.train.num_epochs = args.epochs
    if args.device:
        config.train.device = args.device
    if args.output_dir:
        config.train.output_dir = args.output_dir
        config.train.log_dir = f"{args.output_dir}/logs"
        config.train.checkpoint_dir = f"{args.output_dir}/checkpoints"

    # Validate config
    errors = config.validate()
    if errors:
        logger.error("Configuration errors: %s", errors)
        return None

    # Build dataset
    data_dir = args.data_dir
    if not Path(data_dir).exists():
        logger.error("Data directory not found: %s", data_dir)
        return None

    # Split dataset
    train_files, val_files = split_dataset(
        data_dir,
        train_ratio=args.train_ratio,
        seed=config.train.seed,
    )

    if not train_files:
        logger.error("No training data found in %s", data_dir)
        return None

    # P1-5: Pass file_list to RVCDataset to ensure train/val split is respected
    train_dataset = RVCDataset(data_dir, config.data, file_list=train_files)
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=config.train.pin_memory,
    )

    val_loader = None
    if val_files:
        val_dataset = RVCDataset(data_dir, config.data, file_list=val_files)
        val_loader = create_dataloader(
            val_dataset,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=max(1, config.train.num_workers // 2),
        )

    # Build trainer
    trainer = RVCTrainer(config)
    trainer.build_models()
    trainer.build_optimizers()

    if not args.no_tensorboard:
        trainer.setup_tensorboard()

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        logger.info("Resuming from epoch %d", start_epoch)

    # Save config
    config.save_json(f"{config.train.output_dir}/training_config.json")

    # Train
    try:
        trainer.train(train_loader, val_loader, start_epoch=start_epoch)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        # Save final checkpoint
        trainer.save_checkpoint(trainer.current_epoch + 1)
        trainer.close()

    return trainer


def cmd_export(args):
    """Handle 'export' subcommand."""
    from .config import TrainingConfig
    from .trainer import RVCTrainer

    if not args.checkpoint:
        logger.error("--checkpoint is required for export")
        return None

    # Load config
    if args.config:
        config = TrainingConfig.load_json(args.config)
    else:
        config = TrainingConfig()

    # Determine export format: CLI --format overrides config export_format
    export_format = getattr(args, 'format', None) or config.train.export_format

    # Build trainer and load checkpoint
    trainer = RVCTrainer(config, device=args.device or "cpu")
    trainer.build_models()
    trainer.load_checkpoint(args.checkpoint)

    # Export based on format
    output_dir = args.output or "output/rvc_exported"
    output_path = Path(output_dir)

    if export_format == "rvc":
        rvc_path = output_path.with_suffix(".pt") if output_path.suffix == "" else output_path
        trainer.export_for_inference(str(rvc_path))
        logger.info("RVC model exported to: %s", rvc_path)
    elif export_format == "sovits":
        sovits_path = output_path.with_suffix(".pt") if output_path.suffix == "" else output_path
        trainer.export_sovits_format(str(sovits_path))
        logger.info("SoVITS model exported to: %s", sovits_path)
    elif export_format == "dual":
        output_path.mkdir(parents=True, exist_ok=True)
        result = trainer.export_dual_format(
            output_dir=str(output_path),
            rvc_name="model_rvc.pt",
            sovits_name="model_sovits.pt",
        )
        logger.info("Dual export complete: RVC=%s, SoVITS=%s", result["rvc"], result["sovits"])
    else:
        logger.error("Unknown export format: %s", export_format)
        return None

    trainer.close()
    return output_path


def cmd_separate(args):
    """Handle 'separate' subcommand."""
    from src.separators.audio_separator import AudioSeparator

    logger.info("Separating audio: %s", args.input)
    
    separator = AudioSeparator(
        backend=args.backend,
        device=args.device,
    )
    
    # Perform separation
    if args.mode == "hpss":
        harmonic, percussive = separator.hpss(args.input)
        # Save outputs
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stem_name = Path(args.input).stem
        _save_audio(harmonic, output_dir / f"{stem_name}_harmonic.wav", 44100)
        _save_audio(percussive, output_dir / f"{stem_name}_percussive.wav", 44100)
        
        logger.info("HPSS separation complete:")
        logger.info("  Harmonic: %s", output_dir / f"{stem_name}_harmonic.wav")
        logger.info("  Percussive: %s", output_dir / f"{stem_name}_percussive.wav")
        
    elif args.mode == "dereverb":
        dry = separator.dereverb(args.input, method=args.dereverb_method)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stem_name = Path(args.input).stem
        _save_audio(dry, output_dir / f"{stem_name}_dry.wav", 44100)
        
        logger.info("Dereverberation complete:")
        logger.info("  Output: %s", output_dir / f"{stem_name}_dry.wav")
        
    else:
        # 2-stem or 4-stem separation
        result = separator.separate(args.input, mode=args.mode)
        
        # Save outputs
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stem_name = Path(args.input).stem
        
        if args.mode == "2stems":
            vocals, accompaniment = result
            _save_audio(vocals, output_dir / f"{stem_name}_vocals.wav", 44100)
            _save_audio(accompaniment, output_dir / f"{stem_name}_accompaniment.wav", 44100)
            logger.info("2-stem separation complete:")
            logger.info("  Vocals: %s", output_dir / f"{stem_name}_vocals.wav")
            logger.info("  Accompaniment: %s", output_dir / f"{stem_name}_accompaniment.wav")
        else:  # 4stems
            vocals, drums, bass, other = result
            _save_audio(vocals, output_dir / f"{stem_name}_vocals.wav", 44100)
            _save_audio(drums, output_dir / f"{stem_name}_drums.wav", 44100)
            _save_audio(bass, output_dir / f"{stem_name}_bass.wav", 44100)
            _save_audio(other, output_dir / f"{stem_name}_other.wav", 44100)
            logger.info("4-stem separation complete:")
            logger.info("  Vocals: %s", output_dir / f"{stem_name}_vocals.wav")
            logger.info("  Drums: %s", output_dir / f"{stem_name}_drums.wav")
            logger.info("  Bass: %s", output_dir / f"{stem_name}_bass.wav")
            logger.info("  Other: %s", output_dir / f"{stem_name}_other.wav")


def cmd_cover(args):
    """Handle 'cover' subcommand."""
    from .cover_pipeline import CoverPipeline, CoverConfig

    logger.info("Generating AI cover: %s", args.input)
    
    config = CoverConfig(
        separate_vocals=not args.no_separate,
        separation_mode=args.mode,
        separation_backend=args.backend,
        dereverb=args.dereverb,
        transpose=args.transpose,
        mix_with_accompaniment=not args.no_mix,
        vocal_volume=args.vocal_volume,
        accompaniment_volume=args.accompaniment_volume,
    )
    
    pipeline = CoverPipeline(
        model_path=args.model,
        config=config,
        device=args.device,
    )
    
    result = pipeline.generate_cover(
        source_audio=args.input,
        output_path=args.output,
    )
    
    logger.info("Cover generation complete!")
    logger.info("  Output: %s", result.output_path)
    logger.info("  Sample rate: %d Hz", result.sample_rate)


def _save_audio(audio, path, sample_rate):
    """Helper to save audio to file."""
    import numpy as np
    from scipy.io import wavfile
    
    # Convert to 16-bit PCM
    if audio.dtype != np.int16:
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)
    
    wavfile.write(str(path), sample_rate, audio)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        prog="soma-train",
        description="SOMA RVC Voice Conversion Model Training Tool",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose (debug) logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- preprocess ---
    p_preprocess = subparsers.add_parser(
        "preprocess",
        help="Preprocess audio files for training",
    )
    p_preprocess.add_argument(
        "input_dir", type=str,
        help="Input directory containing audio files",
    )
    p_preprocess.add_argument(
        "output_dir", type=str,
        help="Output directory for processed segments",
    )
    p_preprocess.add_argument(
        "--config", type=str, default=None,
        help="Path to training config JSON file",
    )
    p_preprocess.add_argument(
        "--sample-rate", type=int, default=40000,
        help="Target sample rate (default: 40000)",
    )
    p_preprocess.add_argument(
        "--segment-duration", type=float, default=3.0,
        help="Segment duration in seconds (default: 3.0)",
    )
    p_preprocess.add_argument(
        "--overlap", type=float, default=0.1,
        help="Overlap ratio between segments (default: 0.1)",
    )
    p_preprocess.add_argument(
        "--silence-threshold", type=float, default=-40.0,
        help="Silence threshold in dB (default: -40.0)",
    )
    p_preprocess.add_argument(
        "--normalize-mode", type=str, choices=["peak", "lufs"], default="peak",
        help="Normalization mode (default: peak)",
    )
    p_preprocess.add_argument(
        "--no-recursive", dest="recursive", action="store_false",
        help="Do not search subdirectories",
    )
    p_preprocess.set_defaults(recursive=True)

    # --- train ---
    p_train = subparsers.add_parser(
        "train",
        help="Train RVC model",
    )
    p_train.add_argument(
        "data_dir", type=str,
        help="Directory containing preprocessed .npy segments",
    )
    p_train.add_argument(
        "--config", type=str, default=None,
        help="Path to training config JSON file",
    )
    p_train.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size",
    )
    p_train.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs",
    )
    p_train.add_argument(
        "--device", type=str, default=None,
        help="Override device (cpu/cuda/mps)",
    )
    p_train.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory",
    )
    p_train.add_argument(
        "--train-ratio", type=float, default=0.9,
        help="Train/validation split ratio (default: 0.9)",
    )
    p_train.add_argument(
        "--resume", type=str, default=None,
        help="Resume from checkpoint file",
    )
    p_train.add_argument(
        "--no-tensorboard", action="store_true",
        help="Disable TensorBoard logging",
    )

    # --- export ---
    p_export = subparsers.add_parser(
        "export",
        help="Export trained model for inference",
    )
    p_export.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to training checkpoint (.pt)",
    )
    p_export.add_argument(
        "--output", type=str, default=None,
        help="Output path for exported model (default: output/rvc_exported.pt)",
    )
    p_export.add_argument(
        "--config", type=str, default=None,
        help="Path to training config JSON file",
    )
    p_export.add_argument(
        "--device", type=str, default="cpu",
        help="Device for export (default: cpu)",
    )
    p_export.add_argument(
        "--format", type=str, choices=["rvc", "sovits", "dual"], default=None,
        help="Export format: 'rvc' (RVC only), 'sovits' (SoVITS only), 'dual' (both). "
             "Defaults to config export_format or 'rvc' if not set.",
    )

    # --- separate ---
    p_separate = subparsers.add_parser(
        "separate",
        help="Separate audio into stems (vocals, accompaniment, etc.)",
    )
    p_separate.add_argument(
        "input", type=str,
        help="Input audio file path",
    )
    p_separate.add_argument(
        "--mode", type=str, choices=["2stems", "4stems", "hpss", "dereverb"], default="2stems",
        help="Separation mode: '2stems' (vocals+accompaniment), '4stems' (vocals+drums+bass+other), "
             "'hpss' (harmonic+percussive), 'dereverb' (remove reverb)",
    )
    p_separate.add_argument(
        "--backend", type=str, choices=["demucs", "msst", "librosa"], default="demucs",
        help="Separation backend (default: demucs)",
    )
    p_separate.add_argument(
        "--output-dir", type=str, default="output/separated",
        help="Output directory for separated stems (default: output/separated)",
    )
    p_separate.add_argument(
        "--dereverb-method", type=str, choices=["spectral", "wiener"], default="spectral",
        help="Dereverberation method (default: spectral)",
    )
    p_separate.add_argument(
        "--device", type=str, default=None,
        help="Device for separation (cpu/cuda/mps)",
    )

    # --- cover ---
    p_cover = subparsers.add_parser(
        "cover",
        help="Generate AI cover of a song",
    )
    p_cover.add_argument(
        "input", type=str,
        help="Input audio file path",
    )
    p_cover.add_argument(
        "--model", type=str, required=True,
        help="Path to trained RVC model (.pth)",
    )
    p_cover.add_argument(
        "--output", type=str, default="output/cover.wav",
        help="Output path for generated cover (default: output/cover.wav)",
    )
    p_cover.add_argument(
        "--mode", type=str, choices=["2stems", "4stems"], default="2stems",
        help="Separation mode for vocals extraction (default: 2stems)",
    )
    p_cover.add_argument(
        "--backend", type=str, choices=["demucs", "msst", "librosa"], default="demucs",
        help="Separation backend (default: demucs)",
    )
    p_cover.add_argument(
        "--transpose", type=int, default=0,
        help="Pitch shift in semitones (default: 0)",
    )
    p_cover.add_argument(
        "--dereverb", action="store_true",
        help="Apply dereverberation to extracted vocals",
    )
    p_cover.add_argument(
        "--no-separate", action="store_true",
        help="Skip vocals separation (input is already vocals)",
    )
    p_cover.add_argument(
        "--no-mix", action="store_true",
        help="Output only converted vocals, no mixing with accompaniment",
    )
    p_cover.add_argument(
        "--vocal-volume", type=float, default=1.0,
        help="Volume of converted vocals (default: 1.0)",
    )
    p_cover.add_argument(
        "--accompaniment-volume", type=float, default=0.8,
        help="Volume of accompaniment (default: 0.8)",
    )
    p_cover.add_argument(
        "--device", type=str, default=None,
        help="Device for inference (cpu/cuda/mps)",
    )

    return parser


def main(argv=None):
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "preprocess": cmd_preprocess,
        "train": cmd_train,
        "export": cmd_export,
        "separate": cmd_separate,
        "cover": cmd_cover,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
