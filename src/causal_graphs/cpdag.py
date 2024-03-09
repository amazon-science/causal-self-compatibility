from __future__ import annotations

import io
import logging
from typing import List, Iterable, Any, Union, Dict, Set

import networkx as nx
import numpy as np
import pydot
from causallearn.graph.Dag import Dag
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.DAG2CPDAG import dag2cpdag
from cdt.metrics import SHD
from matplotlib import image as mpimg

from src.causal_graphs.sc_causal_graph import SCCausalGraph
from src.metrics import edge_precision, edge_recall, edge_f1, edge_fpr, skeleton_tpr, skeleton_fpr, \
    skeleton_precision, skeleton_f1, avg_gt_degree, avg_degree


class CPDAG(SCCausalGraph):

    def __init__(self, graph: Union[nx.DiGraph, GeneralGraph]):
        if type(graph) == nx.DiGraph:
            super().__init__(graph)
        elif type(graph) == GeneralGraph:
            super().__init__(self._cl_cpdag_to_nx(graph))
        else:
            raise TypeError()

    def edge_precision(self, ground_truth: nx.DiGraph) -> float:
        return edge_precision(self._dag_to_cpdag(ground_truth), self.graph)

    def edge_recall(self, ground_truth: nx.DiGraph) -> float:
        return edge_recall(self._dag_to_cpdag(ground_truth), self.graph)

    def edge_f1(self, ground_truth: nx.DiGraph) -> float:
        return edge_f1(self._dag_to_cpdag(ground_truth), self.graph)

    def edge_fpr(self, ground_truth: nx.DiGraph) -> float:
        return edge_fpr(self._dag_to_cpdag(ground_truth), self.graph)

    def skeleton_tpr(self, ground_truth: nx.DiGraph) -> float:
        return skeleton_tpr(ground_truth, self.graph)

    def skeleton_fpr(self, ground_truth: nx.DiGraph) -> float:
        return skeleton_fpr(ground_truth, self.graph)

    def skeleton_precision(self, ground_truth: nx.DiGraph) -> float:
        return skeleton_precision(ground_truth, self.graph)

    def skeleton_f1(self, ground_truth: nx.DiGraph) -> float:
        return skeleton_f1(ground_truth, self.graph)

    def avg_gt_degree(self, ground_truth: nx.DiGraph) -> float:
        return avg_gt_degree(ground_truth, self.graph)

    def avg_degree(self, ground_truth: nx.DiGraph) -> float:
        return avg_degree(ground_truth, self.graph)

    def shd(self, ground_truth: Union[nx.DiGraph, SCCausalGraph]) -> int:
        if type(ground_truth) == nx.DiGraph:
            return SHD(self._dag_to_cpdag(ground_truth), self.graph)
        elif type(ground_truth) == CPDAG:
            return SHD(ground_truth.graph, self.graph)
        else:
            raise NotImplementedError()

    @staticmethod
    def valid_metrics() -> List[str]:
        return ['edge_precision', 'edge_recall', 'edge_f1', 'edge_fpr', 'shd', 'skeleton_f1', 'avg_degree',
                'avg_gt_degree', 'skeleton_tpr', 'skeleton_fpr', 'skeleton_precision']

    def variables(self) -> List[Any]:
        return list(self.graph.nodes)

    def get_adjustment_set(self, x: Any, y: Any) -> Set[Any]:
        return set(self.graph.predecessors(x))

    def marginalize(self, remaining_nodes: Iterable[Any]) -> CPDAG:
        g_marginalised = self.graph.copy()
        subset = set(remaining_nodes)
        for node in set(g_marginalised.nodes) - subset:
            self._marginalise_node(g_marginalised, node)
        return CPDAG(g_marginalised)

    def adjustment_valid(self, adjustment_set: Set[Any], x: Any, y: Any) -> bool:
        for parent in self.graph.predecessors(x):
            if self.graph.has_edge(x, parent):  # If bidirected edge between parent and x, the adjustment is not valid
                return False
        return True

    def save_graph(self, graph_dir: str):
        nx.write_graphml(self.graph, graph_dir)

    def is_sufficient(self, nodes: Iterable[Any]) -> bool:
        logging.warning('Sufficiency test for CPDAGs is experimental and untested.')
        marginal_graph = self.graph.copy()
        remaining_nodes = set(nodes)
        # all nodes that ought to be removed
        for node in set(self.variables()).difference(remaining_nodes):
            # Add edge, if node is mediator
            for pre in marginal_graph.predecessors(node):
                for succ in marginal_graph.successors(node):
                    if pre != succ:
                        marginal_graph.add_edge(pre, succ)
                # not sufficient if node is possible confounder of nodes
                for suc_one in marginal_graph.successors(node):
                    for suc_two in marginal_graph.successors(node):
                        if suc_two != suc_one:
                            if suc_one in remaining_nodes and suc_two in remaining_nodes:
                                return False
                            else:
                                marginal_graph.add_edge(suc_one, suc_two)
                                marginal_graph.add_edge(suc_two, suc_one)
            marginal_graph.remove_node(node)
        return True

    def get_permuted_graph(self) -> CPDAG:
        node_permutation = {n: n_new for (n, n_new) in zip(self.graph.nodes, np.random.permutation(self.graph.nodes))}
        return CPDAG(nx.relabel_nodes(self.graph, node_permutation, copy=True))

    @staticmethod
    def _cl_cpdag_to_nx(graph: GeneralGraph) -> nx.DiGraph:
        g_hat = nx.DiGraph()
        g_hat.add_nodes_from([node.get_name() for node in graph.get_nodes()])
        for edge in graph.get_graph_edges():
            if edge.get_endpoint1() == Endpoint.ARROW:
                g_hat.add_edge(edge.get_node2().get_name(), edge.get_node1().get_name())
            if edge.get_endpoint2() == Endpoint.ARROW:
                g_hat.add_edge(edge.get_node1().get_name(), edge.get_node2().get_name())
            if edge.get_endpoint1() == Endpoint.CIRCLE or edge.get_endpoint2() == Endpoint.CIRCLE:
                raise ValueError('Did not expect edge with circle')
            if edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.TAIL:
                g_hat.add_edge(edge.get_node2().get_name(), edge.get_node1().get_name())
                g_hat.add_edge(edge.get_node1().get_name(), edge.get_node2().get_name())
        return g_hat

    @staticmethod
    def _dag_to_cpdag(graph: nx.DiGraph) -> nx.DiGraph:
        if not nx.is_directed_acyclic_graph(graph):
            logging.warning('Ground truth is not a DAG!')
            return graph
        nodes_dict = {name: GraphNode(name) for name in graph.nodes}
        dag = Dag(list(nodes_dict.values()))
        for (src, trgt) in graph.edges:
            dag.add_directed_edge(dag.get_node(src), dag.get_node(trgt))
        cpdag = dag2cpdag(dag)
        return CPDAG._cl_cpdag_to_nx(cpdag)

    @staticmethod
    def _marginalise_node(graph: nx.DiGraph, node: Any):
        # Add egdes from all predecessors to sucessors, i.e. replace  x -> node -> y with a new edge x -> y
        for pre in graph.predecessors(node):
            for succ in graph.successors(node):
                if pre != succ:
                    weight = graph[pre][node]['weight'] + graph[node][succ]['weight'] if 'weight' in graph[pre][
                        node] and 'weight' in graph[node][succ] else 2
                    graph.add_edge(pre, succ, weight=weight)

        graph.remove_node(node)

    @staticmethod
    def load_graph(graph_dir: str) -> CPDAG:
        return CPDAG(nx.read_graphml(graph_dir))

    def visualize(self, ax=None):
        title = ""
        dpi: float = 200
        fontsize: int = 18

        pydot_g = pydot.Dot(title, graph_type="digraph", fontsize=fontsize)
        pydot_g.obj_dict["attributes"]["dpi"] = dpi

        for node in self.graph.nodes:
            pydot_g.add_node(pydot.Node(node))

        already_bidirected = set([])
        for node_one, node_two in self.graph.edges:
            if self.graph.has_edge(node_two, node_one):
                if (node_one, node_two) not in already_bidirected:
                    dot_edge = pydot.Edge(node_one, node_two, dir='both', arrowtail='normal', arrowhead='normal')
                    already_bidirected.add((node_two, node_one))
                    pydot_g.add_edge(dot_edge)
            else:
                dot_edge = pydot.Edge(node_one, node_two, arrowtail='none', arrowhead='normal')
                pydot_g.add_edge(dot_edge)

        tmp_png = pydot_g.create_png(f="png")
        fp = io.BytesIO(tmp_png)
        img = mpimg.imread(fp, format='png')
        ax.set_axis_off()
        ax.imshow(img)

    def eval_all_metrics(self, ground_truth: nx.DiGraph) -> Dict[str, float]:
        if len(ground_truth.nodes) > len(self.variables()):  # TODO if-else neccesarry?
            ground_truth_cpdag = CPDAG(ground_truth)
            marginalised_ground_truth = ground_truth_cpdag.marginalize(self.variables())
            return super().eval_all_metrics(marginalised_ground_truth.graph)
        else:
            return super().eval_all_metrics(ground_truth)
