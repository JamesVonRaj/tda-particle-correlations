"""
Pipeline orchestration for TDA analysis.

This module provides the PipelineRunner class that coordinates
execution of all pipeline stages based on configuration.
"""

import json
import logging
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

from . import stages
from .core.progress import format_time


class PipelineRunner:
    """
    Orchestrates execution of the TDA pipeline stages.

    Parameters
    ----------
    config : dict
        Configuration dictionary (from load_config)
    output_dir : str or Path, optional
        Override output directory. If None, creates timestamped folder.
    dry_run : bool
        If True, only print what would be executed without running.
    stages_to_run : list[str], optional
        Subset of stages to run. If None, runs all enabled stages.
    """

    STAGE_ORDER = ['data_generation', 'persistence', 'analysis', 'visualization']

    def __init__(
        self,
        config: dict,
        output_dir: Optional[str | Path] = None,
        dry_run: bool = False,
        stages_to_run: Optional[list[str]] = None
    ):
        self.config = config
        self.dry_run = dry_run
        self.stages_to_run = stages_to_run
        self.logger = None

        # Create output directory
        self.output_dir = self._create_output_dir(output_dir)

        # Track metadata
        self.metadata = {
            'experiment_name': config['experiment']['name'],
            'start_time': None,
            'end_time': None,
            'stages_run': [],
            'errors': [],
            'config_path': config.get('_config_path', 'unknown')
        }

    def _create_output_dir(self, output_dir: Optional[str | Path]) -> Path:
        """Create and return the output directory."""
        if output_dir is not None:
            output_path = Path(output_dir)
        else:
            # Create timestamped folder in outputs/
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_name = self.config['experiment']['name']
            folder_name = f"{experiment_name}_{timestamp}"

            # Find the project root (where outputs/ should be)
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent
            output_path = project_root / 'outputs' / folder_name

        if not self.dry_run:
            output_path.mkdir(parents=True, exist_ok=True)

        return output_path

    def _setup_logging(self) -> logging.Logger:
        """Configure logging for the pipeline."""
        log_level = getattr(logging, self.config['pipeline']['log_level'], logging.INFO)

        # Create logger
        logger = logging.getLogger('tda_pipeline')
        logger.setLevel(log_level)
        logger.handlers = []  # Clear existing handlers

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_format = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

        # File handler (if not dry run)
        if not self.dry_run:
            log_file = self.output_dir / 'run.log'
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
            file_format = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)

        return logger

    def _copy_config(self):
        """Copy configuration to output directory."""
        if self.dry_run:
            return

        # Save config as YAML
        import yaml
        config_copy = {k: v for k, v in self.config.items() if not k.startswith('_')}
        config_file = self.output_dir / 'config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config_copy, f, default_flow_style=False, sort_keys=False)

    def _save_metadata(self):
        """Save run metadata to output directory."""
        if self.dry_run:
            return

        metadata_file = self.output_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def _set_random_seed(self):
        """Set random seed if specified in config."""
        seed = self.config['pipeline'].get('random_seed')
        if seed is not None:
            np.random.seed(seed)
            self.logger.info(f"Random seed set to {seed}")

    def _should_run_stage(self, stage_name: str) -> bool:
        """Determine if a stage should be run."""
        # Check if stage is in subset (if specified)
        if self.stages_to_run is not None:
            if stage_name not in self.stages_to_run:
                return False

        # Check if stage is enabled in config
        stage_config = self.config.get(stage_name, {})
        return stage_config.get('enabled', True)

    def run(self) -> dict:
        """
        Execute the pipeline.

        Returns
        -------
        dict
            Metadata about the run including any errors
        """
        self.logger = self._setup_logging()
        self.metadata['start_time'] = datetime.now().isoformat()
        pipeline_start_time = time.time()

        self.logger.info("=" * 60)
        self.logger.info(f"TDA Pipeline: {self.config['experiment']['name']}")
        self.logger.info("=" * 60)

        if self.dry_run:
            self.logger.info("DRY RUN - No changes will be made")

        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Copy config and set random seed
        self._copy_config()
        self._set_random_seed()

        # Determine which stages will run
        stages_to_run = [s for s in self.STAGE_ORDER if self._should_run_stage(s)]
        total_stages = len(stages_to_run)

        self.logger.info(f"Stages to run: {', '.join(stages_to_run)} ({total_stages} total)")

        # Track stage timings
        stage_timings = {}

        # Run each stage in order
        for stage_idx, stage_name in enumerate(self.STAGE_ORDER):
            if not self._should_run_stage(stage_name):
                self.logger.info(f"\nSkipping stage: {stage_name} (disabled or not selected)")
                continue

            # Calculate progress
            completed_stages = stage_idx
            elapsed = time.time() - pipeline_start_time

            if completed_stages > 0 and completed_stages < total_stages:
                avg_stage_time = elapsed / completed_stages
                remaining_stages = total_stages - completed_stages
                eta_seconds = remaining_stages * avg_stage_time
                eta_time = datetime.now() + timedelta(seconds=eta_seconds)
                eta_str = f"{format_time(eta_seconds)} (est. {eta_time.strftime('%H:%M:%S')})"
            else:
                eta_str = "calculating..."

            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Stage [{stage_idx + 1}/{total_stages}]: {stage_name}")
            self.logger.info(f"Pipeline progress: {completed_stages}/{total_stages} stages | "
                           f"Elapsed: {format_time(elapsed)} | ETA: {eta_str}")
            self.logger.info("=" * 60)

            if self.dry_run:
                self.logger.info(f"Would run: {stage_name}")
                self.metadata['stages_run'].append(stage_name)
                continue

            stage_start_time = time.time()

            try:
                stage_module = getattr(stages, stage_name)
                stage_module.run(self.config, self.output_dir, self.logger)
                self.metadata['stages_run'].append(stage_name)

                stage_elapsed = time.time() - stage_start_time
                stage_timings[stage_name] = stage_elapsed
                self.logger.info(f"Stage {stage_name} completed in {format_time(stage_elapsed)}")

            except Exception as e:
                error_msg = f"Error in stage {stage_name}: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                self.metadata['errors'].append({
                    'stage': stage_name,
                    'error': str(e),
                    'time': datetime.now().isoformat()
                })

                if not self.config['pipeline'].get('continue_on_error', False):
                    self.logger.error("Pipeline stopped due to error")
                    break

        total_elapsed = time.time() - pipeline_start_time
        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['total_time_seconds'] = total_elapsed
        self.metadata['stage_timings'] = stage_timings
        self._save_metadata()

        self.logger.info("\n" + "=" * 60)
        if self.metadata['errors']:
            self.logger.warning(f"Pipeline completed with {len(self.metadata['errors'])} error(s)")
        else:
            self.logger.info("Pipeline completed successfully")

        self.logger.info(f"Total time: {format_time(total_elapsed)}")
        self.logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Output directory: {self.output_dir}")

        # Print stage timing breakdown
        if stage_timings:
            self.logger.info("\nStage timing breakdown:")
            for stage_name, stage_time in stage_timings.items():
                pct = stage_time / total_elapsed * 100 if total_elapsed > 0 else 0
                self.logger.info(f"  {stage_name}: {format_time(stage_time)} ({pct:.1f}%)")

        self.logger.info("=" * 60)

        return self.metadata
