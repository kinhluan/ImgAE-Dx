"""Command-line interface for ImgAE-Dx."""

from .train import train_command
from .evaluate import evaluate_command
from .config import config_command

__all__ = ["train_command", "evaluate_command", "config_command"]