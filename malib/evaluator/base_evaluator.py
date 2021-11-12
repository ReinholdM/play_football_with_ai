from abc import ABCMeta, abstractmethod
from typing import Dict, Any


class BaseEvaluator(metaclass=ABCMeta):
    """Abstract base class for evaluator."""

    def __init__(self, metrics, name="default"):
        self._metrics = metrics

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Dict[str, Any]:
        pass
