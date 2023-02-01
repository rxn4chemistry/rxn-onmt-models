from abc import ABC, abstractmethod
from typing import Any, Dict, Type, TypeVar

from rxn_onmt_utils.rxn_models.metrics_files import MetricsFiles

CalculatorT = TypeVar("CalculatorT", bound="MetricsCalculator")


class MetricsCalculator(ABC):
    """
    Base class for calculating metrics and returning them in a dictionary.
    """

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Calculate the metrics.

        Note: the paths to ground truth and prediction are to be set in the
        constructor of the derived class."""

    @classmethod
    @abstractmethod
    def from_metrics_files(
        cls: Type[CalculatorT], metrics_files: MetricsFiles
    ) -> CalculatorT:
        """Build the instance from the MetricsFiles object."""
