#!/usr/bin/env python
"""
Run TDA Pipeline

Main entry point for running the configuration-driven TDA pipeline.

Usage:
    python scripts/run_pipeline.py config.yaml
    python scripts/run_pipeline.py config.yaml --dry-run
    python scripts/run_pipeline.py config.yaml --stages data_generation persistence
    python scripts/run_pipeline.py config.yaml --output-dir ./my_results

Examples:
    # Run full pipeline
    python scripts/run_pipeline.py experiments/sweep_config.yaml

    # Run only data generation and persistence
    python scripts/run_pipeline.py config.yaml --stages data_generation persistence

    # Dry run to see what would be executed
    python scripts/run_pipeline.py config.yaml --dry-run

    # Specify custom output directory
    python scripts/run_pipeline.py config.yaml --output-dir ./my_results
"""

import argparse
import sys
from pathlib import Path

# Add the scripts directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from pipeline import load_config, validate_config, PipelineRunner
from pipeline.config import get_config_summary


def main():
    parser = argparse.ArgumentParser(
        description='Run TDA analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        'config',
        type=str,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would be executed without running'
    )

    parser.add_argument(
        '--stages',
        nargs='+',
        choices=['data_generation', 'persistence', 'analysis', 'visualization'],
        help='Run only specified stages'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate configuration, do not run'
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate configuration
    errors = validate_config(config)
    if errors:
        print("Configuration validation errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)

    # Print configuration summary
    print("\n" + "=" * 60)
    print("Configuration Summary")
    print("=" * 60)
    print(get_config_summary(config))
    print("=" * 60 + "\n")

    if args.validate_only:
        print("Configuration is valid.")
        sys.exit(0)

    # Create and run pipeline
    runner = PipelineRunner(
        config=config,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        stages_to_run=args.stages
    )

    metadata = runner.run()

    # Exit with error code if there were errors
    if metadata['errors']:
        sys.exit(1)


if __name__ == '__main__':
    main()
