# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import abstractmethod, ABCMeta
from typing import Iterable, Any

import networkx as nx
import pandas as pd
from causallearn.graph.Edge import Edge
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.FCMBased.lingam import RCD
from causallearn.search.ScoreBased.GES import ges

from causal_graphs.admg import ADMG
from causal_graphs.cpdag import CPDAG
from causal_graphs.pag import PAG
from causal_graphs.sc_causal_graph import SCCausalGraph


class DiscoverAlgorithm(metaclass=ABCMeta):
    """
    Base class of causal discovery algorithms.
    """

    @abstractmethod
    def __call__(self, data: pd.DataFrame) -> SCCausalGraph:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def load_graph(graph_dir: str) -> SCCausalGraph:
        raise NotImplementedError()


class RCDLINGAM(DiscoverAlgorithm):
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def __call__(self, data: pd.DataFrame) -> ADMG:
        lingam = RCD(cor_alpha=self.alpha, ind_alpha=self.alpha, shapiro_alpha=self.alpha)
        lingam.fit(data)
        g_hat = nx.from_numpy_array(lingam.adjacency_matrix_.T, create_using=nx.DiGraph)
        nx.relabel_nodes(g_hat, {i: name for (i, name) in enumerate(data.keys())}, copy=False)
        return ADMG(g_hat)

    @staticmethod
    def load_graph(graph_dir: str) -> ADMG:
        return ADMG.load_graph(graph_dir)


class FCI(DiscoverAlgorithm):
    def __init__(self, alpha: float = 0.01, indep_test: str = 'fisherz'):
        self.alpha = alpha
        self.indep_test = indep_test

    def __call__(self, data: pd.DataFrame) -> PAG:
        result = fci(data.to_numpy(), alpha=self.alpha, independence_test_method=self.indep_test)[0]
        name_map = {'X{}'.format(i + 1): name for (i, name) in enumerate(data.keys())}
        nodes = [GraphNode(name) for name in data.keys()]
        pag = GeneralGraph(nodes)
        for edge in result.get_graph_edges():
            e_one = pag.get_node(name_map[edge.get_node1().get_name()])
            e_two = pag.get_node(name_map[edge.get_node2().get_name()])
            new_edge = Edge(e_one, e_two, edge.get_endpoint1(), edge.get_endpoint2())
            pag.add_edge(new_edge)
        return PAG(pag)

    @staticmethod
    def load_graph(graph_dir: str) -> PAG:
        return PAG.load_graph(graph_dir)


class PC(DiscoverAlgorithm):
    def __init__(self, alpha: float = 0.001, indep_test: str = 'fisherz', tunable_param: str = 'alpha',
                 tunable_param_values: Iterable[Any] = None):
        self.alpha = alpha
        self.indep_test = indep_test
        self.tunable_param = tunable_param  # For cross validation
        if tunable_param_values is None:
            self.tunable_param_values = [.1, .01, .001, .0001]

    def __call__(self, data: pd.DataFrame) -> CPDAG:
        pc_result = pc(data.to_numpy(), alpha=self.alpha, node_names=list(data.keys()), indep_test=self.indep_test)
        return CPDAG(pc_result.G)

    @staticmethod
    def load_graph(graph_dir: str) -> CPDAG:
        return CPDAG.load_graph(graph_dir)


class GES(DiscoverAlgorithm):
    def __init__(self, lambda_param: float = 2, tunable_param: str = 'lambda_param',
                 tunable_param_values: Iterable[Any] = None):
        self.lambda_param = lambda_param
        self.tunable_param = tunable_param  # For cross validation
        if tunable_param_values is None:
            self.tunable_param_values = [.1, .01, .001, .0001]

    def __call__(self, data: pd.DataFrame) -> CPDAG:
        ges_res = ges(data.to_numpy(), parameters={'lambda': self.lambda_param})
        g_hat = nx.DiGraph()
        g_hat.add_nodes_from(data.keys())
        for (i, x), (j, y) in [((i, x), (j, y)) for (i, x) in enumerate(data.keys()) for (j, y) in
                               enumerate(data.keys())
                               if x != y]:
            if ges_res['G'].graph[i, j] == -1 and ges_res['G'].graph[j, i] == 1:
                g_hat.add_edge(x, y)
            elif ges_res['G'].graph[i, j] == -1 and ges_res['G'].graph[i, j] == -1:
                g_hat.add_edge(x, y)  # We add bidirected arrows for undirected edges.
                g_hat.add_edge(y, x)
        return CPDAG(g_hat)

    @staticmethod
    def load_graph(graph_dir: str) -> CPDAG:
        return CPDAG.load_graph(graph_dir)
