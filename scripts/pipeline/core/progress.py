"""
Progress tracking utilities for TDA pipeline.

Provides progress bars with ETA, elapsed time, and completion percentage.
"""

import time
from datetime import datetime, timedelta
from typing import Optional, Iterable, Any
import sys


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 0:
        return "--:--"

    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs:02d}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes:02d}m"


def format_datetime(dt: datetime) -> str:
    """Format datetime for display."""
    return dt.strftime("%H:%M:%S")


class ProgressTracker:
    """
    Progress tracker with ETA calculation.

    Parameters
    ----------
    total : int
        Total number of items to process
    desc : str
        Description for the progress bar
    logger : logging.Logger, optional
        Logger for output (if None, prints to stdout)
    show_bar : bool
        Whether to show a progress bar
    bar_width : int
        Width of the progress bar
    """

    def __init__(
        self,
        total: int,
        desc: str = "Processing",
        logger=None,
        show_bar: bool = True,
        bar_width: int = 30
    ):
        self.total = total
        self.desc = desc
        self.logger = logger
        self.show_bar = show_bar
        self.bar_width = bar_width

        self.current = 0
        self.start_time = None
        self.last_update_time = None
        self.update_interval = 0.5  # Minimum seconds between updates

    def start(self):
        """Start the progress tracker."""
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.current = 0
        self._print_progress()

    def update(self, n: int = 1):
        """Update progress by n items."""
        self.current += n

        # Rate-limit updates to avoid flooding output
        now = time.time()
        if now - self.last_update_time >= self.update_interval or self.current >= self.total:
            self._print_progress()
            self.last_update_time = now

    def _print_progress(self):
        """Print current progress."""
        if self.total == 0:
            return

        elapsed = time.time() - self.start_time if self.start_time else 0
        percent = self.current / self.total * 100

        # Calculate ETA
        if self.current > 0 and self.current < self.total:
            rate = self.current / elapsed
            remaining = (self.total - self.current) / rate
            eta_str = format_time(remaining)
            eta_time = datetime.now() + timedelta(seconds=remaining)
            eta_clock = format_datetime(eta_time)
        elif self.current >= self.total:
            eta_str = "Done"
            eta_clock = format_datetime(datetime.now())
        else:
            eta_str = "--:--"
            eta_clock = "--:--"

        elapsed_str = format_time(elapsed)

        if self.show_bar:
            # Build progress bar
            filled = int(self.bar_width * self.current / self.total)
            bar = "█" * filled + "░" * (self.bar_width - filled)

            msg = (f"\r{self.desc}: |{bar}| {self.current}/{self.total} "
                   f"({percent:.1f}%) | Elapsed: {elapsed_str} | ETA: {eta_str} ({eta_clock})")
        else:
            msg = (f"{self.desc}: {self.current}/{self.total} ({percent:.1f}%) | "
                   f"Elapsed: {elapsed_str} | ETA: {eta_str} ({eta_clock})")

        if self.logger:
            # For logger, only log at certain milestones or completion
            if self.current >= self.total or self.current == 1:
                self.logger.info(msg.strip())
        else:
            # Print to stdout with carriage return for in-place update
            print(msg, end='', flush=True)
            if self.current >= self.total:
                print()  # Newline at completion

    def finish(self):
        """Finish the progress tracker."""
        self.current = self.total
        self._print_progress()
        if not self.logger:
            print()  # Ensure newline


class StageProgressTracker:
    """
    Track progress across multiple stages/configurations.

    Parameters
    ----------
    stages : list
        List of stage names or (name, item_count) tuples
    desc : str
        Overall description
    logger : logging.Logger, optional
        Logger for output
    """

    def __init__(self, stages: list, desc: str = "Pipeline", logger=None):
        self.stages = stages
        self.desc = desc
        self.logger = logger

        # Calculate total items if provided
        if stages and isinstance(stages[0], tuple):
            self.stage_names = [s[0] for s in stages]
            self.stage_counts = [s[1] for s in stages]
            self.total_items = sum(self.stage_counts)
        else:
            self.stage_names = stages
            self.stage_counts = [1] * len(stages)
            self.total_items = len(stages)

        self.current_stage_idx = 0
        self.completed_items = 0
        self.start_time = None
        self.stage_start_time = None

    def start(self):
        """Start tracking."""
        self.start_time = time.time()
        self._log_start()

    def _log_start(self):
        """Log the start of tracking."""
        msg = f"\n{'='*60}\n{self.desc} Progress\n{'='*60}"
        msg += f"\nTotal stages: {len(self.stage_names)}"
        if self.total_items != len(self.stage_names):
            msg += f" | Total items: {self.total_items}"
        msg += f"\nStarted at: {format_datetime(datetime.now())}\n"

        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def start_stage(self, stage_name: str = None):
        """Mark the start of a stage."""
        self.stage_start_time = time.time()

        if stage_name is None and self.current_stage_idx < len(self.stage_names):
            stage_name = self.stage_names[self.current_stage_idx]

        elapsed = time.time() - self.start_time if self.start_time else 0
        percent = self.completed_items / self.total_items * 100 if self.total_items > 0 else 0

        msg = (f"\n[{self.current_stage_idx + 1}/{len(self.stage_names)}] "
               f"Starting: {stage_name} | Overall: {percent:.1f}% | "
               f"Elapsed: {format_time(elapsed)}")

        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def complete_stage(self, items_processed: int = None):
        """Mark a stage as complete."""
        stage_elapsed = time.time() - self.stage_start_time if self.stage_start_time else 0

        if items_processed is None:
            items_processed = self.stage_counts[self.current_stage_idx] if self.current_stage_idx < len(self.stage_counts) else 1

        self.completed_items += items_processed
        self.current_stage_idx += 1

        total_elapsed = time.time() - self.start_time
        percent = self.completed_items / self.total_items * 100 if self.total_items > 0 else 0

        # Calculate ETA
        if self.completed_items > 0 and self.completed_items < self.total_items:
            rate = self.completed_items / total_elapsed
            remaining = (self.total_items - self.completed_items) / rate
            eta_str = format_time(remaining)
            eta_time = datetime.now() + timedelta(seconds=remaining)
        else:
            eta_str = "Done"
            eta_time = datetime.now()

        msg = (f"  Completed in {format_time(stage_elapsed)} | "
               f"Overall: {percent:.1f}% | ETA: {eta_str} ({format_datetime(eta_time)})")

        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def finish(self):
        """Finish tracking and print summary."""
        total_elapsed = time.time() - self.start_time if self.start_time else 0

        msg = f"\n{'='*60}\n{self.desc} Complete\n{'='*60}"
        msg += f"\nTotal time: {format_time(total_elapsed)}"
        msg += f"\nCompleted at: {format_datetime(datetime.now())}\n"

        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)


def progress_iterator(
    iterable: Iterable,
    total: int = None,
    desc: str = "Processing",
    logger=None
) -> Iterable:
    """
    Wrap an iterable with progress tracking.

    Parameters
    ----------
    iterable : Iterable
        Items to iterate over
    total : int, optional
        Total count (inferred from iterable if possible)
    desc : str
        Description
    logger : logging.Logger, optional
        Logger for output

    Yields
    ------
    Items from the iterable
    """
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = 0

    tracker = ProgressTracker(total, desc, logger, show_bar=logger is None)
    tracker.start()

    for item in iterable:
        yield item
        tracker.update(1)

    tracker.finish()
