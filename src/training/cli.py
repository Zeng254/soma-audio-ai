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

    # Create datasets and dataloaders
    train_dataset = RVCDataset(data_dir, config.data)
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=config.train.pin_memory,
    )

    val_loader = None
    if val_files:
        val_dataset = RVCDataset(data_dir, config.data)
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

    # Build trainer and load checkpoint
    trainer = RVCTrainer(config, device=args.device or "cpu")
    trainer.build_models()
    trainer.load_checkpoint(args.checkpoint)

    # Export
    output_path = args.output or "output/rvc_exported.pt"
    trainer.export_for_inference(output_path)
    logger.info("Model exported to: %s", output_path)

    trainer.close()
    return output_path


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
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
