from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import List, Iterable, Any, Tuple, Dict, Set

import networkx as nx


class SCCausalGraph(metaclass=ABCMeta):
    """
    Base class for the causal graphs. The graphs are supposed to be thin wrappers around Networkx graphs or the PAGs
    from Causallearn and provide a unified interface, as well as functionality and metrics required for the
    self-compatibility test.
    """

    def __init__(self, graph: Any):
        self.graph = graph

    @abstractmethod
    def save_graph(self, graph_dir: str):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def load_graph(graph_dir: str) -> SCCausalGraph:
        raise NotImplementedError()

    @staticmethod
    def valid_metrics() -> List[str]:
        return []

    def eval_all_metrics(self, ground_truth: nx.DiGraph) -> Dict[str, float]:
        results = {}
        for metric in self.valid_metrics():
            results[metric] = getattr(self, metric)(ground_truth)
        return results

    @abstractmethod
    def marginalize(self, remaining_nodes: Iterable[Any]):
        raise NotImplementedError()

    def adjustment_valid(self, adjustment_set: Set[Any], x: Any, y: Any) -> bool:
        raise NotImplementedError()

    def get_adjustment_set(self, x: Any, y: Any) -> Set[Any]:
        raise NotImplementedError()

    def shd(self, ground_truth: Tuple[SCCausalGraph, nx.DiGraph]):
        raise NotImplementedError()

    @abstractmethod
    def variables(self) -> List[Any]:
        raise NotImplementedError()

    def get_permuted_graph(self) -> SCCausalGraph:
        raise NotImplementedError()

    @abstractmethod
    def visualize(self, ax=None):
        raise NotImplementedError()
