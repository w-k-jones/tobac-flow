"""
Abstract base classes.
"""

from abc import ABC, abstractmethod
from typing import Callable
import numpy as np


class AbstractFlow(ABC):
    @abstractmethod
    def __init__(
        self, forward_flow: np.ndarray[float], backward_flow: np.ndarray[float]
    ) -> None:
        pass

    @property
    @abstractmethod
    def flow(self) -> tuple[np.ndarray[float], np.ndarray[float]]:
        pass

    @abstractmethod
    def __getitem__(self, items: tuple) -> "AbstractFlow":
        pass

    @abstractmethod
    def convolve(
        self,
        data: np.ndarray[float],
        structure: np.ndarray[bool] = None,
        method: str = "",
        fill_value: float = np.nan,
        dtype: type = np.float32,
        func: Callable | None = None,
    ) -> np.ndarray[float]:
        pass

    @abstractmethod
    def diff(
        self, data: np.ndarray[float], method: str = "linear", dtype: type = np.float32
    ) -> np.ndarray[float]:
        pass

    @abstractmethod
    def sobel(
        self,
        data: np.ndarray[float],
        method: str = "linear",
        dtype: type = None,
        fill_value: float = np.nan,
        direction: str | None = None,
    ) -> np.ndarray[float]:
        pass

    @abstractmethod
    def watershed(
        self,
        field: np.ndarray[float],
        markers: np.ndarray[int],
        mask: np.ndarray[bool] | None = None,
        structure: np.ndarray[bool] = None,
    ) -> np.ndarray[int]:
        pass

    @abstractmethod
    def label(
        self,
        data: np.ndarray[int],
        structure: np.ndarray[bool] = None,
        dtype: type = np.int32,
        overlap: float = 0,
        subsegment_shrink: float = 0,
    ) -> np.ndarray[int]:
        pass

    @abstractmethod
    def link_overlap(
        self,
        data: np.ndarray[bool],
        structure: np.ndarray[bool] = None,
        dtype: type = np.int32,
        overlap: float = 0,
    ) -> np.ndarray[int]:
        pass


__all__ = ("AbstractFlow",)
