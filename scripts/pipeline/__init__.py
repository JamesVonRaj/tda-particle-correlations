"""
Configuration-Driven TDA Pipeline

This module provides a unified pipeline for running TDA analysis on
Thomas Cluster Process data, driven by a YAML configuration file.
"""

from .config import load_config, validate_config
from .runner import PipelineRunner

__all__ = ['load_config', 'validate_config', 'PipelineRunner']
