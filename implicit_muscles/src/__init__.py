"""Core controllers and network implementations for implicit muscles."""

from .network import NNController, WaveController
from .torque_controller import EkebergMuscleController

__all__ = [
    "NNController",
    "WaveController",
    "EkebergMuscleController",
]
