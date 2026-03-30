from abc import ABC, abstractmethod
from typing import Sequence, Optional, Tuple, Any
import numpy as np

  # -------------------------------
  # Base Abstract Interface
  # -------------------------------

class Digitizer(ABC):
    """Abstract base class for all digitizers.        
    Subclasses SHOULD implement the abstract methods and MAY override the optional ones.
    """

    # ---- lifecycle ----
    @abstractmethod
    def connect(self, address: str | None = None) -> None:
        """Connect to the instrument (or get ready).

        Args:
        address: VISA/TCPIP/IP/etc. If the device uses a fixed address, this may be ignored.

        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect and free resources."""
        pass

    # ---- configuration ----
    @abstractmethod
    def set_trigger(self, trigger_type: str = "software", **kwargs) -> None:
        """Configure trigger source and params (driver-specific kwargs allowed)."""
        pass

    # ---- acquisition ----
    @abstractmethod
    def acquire(
        self,
        channels: int | Sequence[int] | None = None,
        num_points: int | None = None,
        sample_rate: float | None = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray|None, np.ndarray|None]:
        """Acquire waveform(s).

        Returns:

            (t, y):

                t: time axis in seconds (shape [N]) or None if unavailable

                y: data array, shape [C, N] or [N] (single channel), or None if unavailable

        """
        pass

    @abstractmethod
    def save_data(self, filename: str, folder: str = ".", data_set_name : str | None = None) -> None:
        """Persist last acquisition to disk (implementation-specific)."""
        pass

    # optional extras
    def acquire_segmented(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def show_plot(self) -> None:
        raise NotImplementedError

    def set_channel(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def set_amplitude(self, *args, **kwargs) -> None:
        raise NotImplementedError