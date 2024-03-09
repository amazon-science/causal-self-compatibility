from __future__ import annotations

import io
from typing import List, Iterable, Any, Union, Dict, Set, Tuple

import networkx as nx
import numpy as np
import pydot
from cdt.metrics import SHD
from matplotlib import image as mpimg

from src.causal_graphs.sc_causal_graph import SCCausalGraph
from src.metrics import edge_precision, edge_recall, edge_f1, edge_fpr, skeleton_tpr, skeleton_fpr, \
    skeleton_precision, skeleton_f1, avg_gt_degree, avg_degree


class ADMG(SCCausalGraph):

    def __init__(self, graph: nx.DiGraph):
        super().__init__(graph)

    def edge_precision(self, ground_truth: nx.DiGraph) -> float:
        return edge_precision(ground_truth, self.graph)

    def edge_recall(self, ground_truth: nx.DiGraph) -> float:
        return edge_recall(ground_truth, self.graph)

    def edge_f1(self, ground_truth: nx.DiGraph) -> float:
        return edge_f1(ground_truth, self.graph)

    def edge_fpr(self, ground_truth: nx.DiGraph) -> float:
        return edge_fpr(ground_truth, self.graph)

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
            return SHD(ground_truth, self.graph)
        elif type(ground_truth) == ADMG:
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

    def adjustment_valid(self, adjustment_set: Set[Any], x: Any, y: Any) -> bool:
        for parent in self.graph.predecessors(x):
            if self.graph.has_edge(x, parent):  # If bidirected edge between parent and x, the adjustment is not valid
                return False
        return True

    def _collect_possible_descendants(self, node: Any, descendants: Set[Any] = None) -> Set[Any]:
        if descendants is None:
            descendants = {node}
        if node in descendants:
            return descendants
        descendants = descendants.union({node})
        possible_children = self.graph.get_successors(node)
        for child in possible_children:
            descendants = descendants.union(self._collect_possible_descendants(child, descendants))
        return descendants

    def _collect_possible_ancestors(self, node: Any, ancestors: Set[Any] = None) -> Set[Any]:
        if ancestors is None:
            ancestors = {node}
        if node in ancestors:
            return ancestors
        ancestors = ancestors.union({node})
        possible_parents = self.graph.get_predecessors(node)
        for parent in possible_parents:
            if self.graph.has_edge(parent, node):
                ancestors = ancestors.union(self._collect_possible_ancestors(parent, ancestors))
        return ancestors

    def marginalize(self, remaining_nodes: Iterable[Any]) -> ADMG:
        g_marginalised = self.graph.copy()
        subset = set(remaining_nodes)
        # Bidirected edges from RCD will already be dealt with by the logic in _marginalise_node().
        # This set is only to ensure that we do not unnecessarily 'lose' orientation information during marginalising
        # one node after the other
        confounded_nodes = set([])
        for node in set(g_marginalised.nodes) - subset:
            g_marginalised, confounded_nodes = self._marginalise_node(g_marginalised, node, confounded_nodes)

        for x, y in confounded_nodes:
            # Add bidirectional edge for confounded nodes
            if x in remaining_nodes and y in remaining_nodes:
                g_marginalised.add_edge(x, y)
                g_marginalised.add_edge(y, x)
        return ADMG(g_marginalised)

    def save_graph(self, graph_dir: str):
        nx.write_graphml(self.graph, graph_dir)

    def is_sufficient(self, nodes: Iterable[Any]) -> bool:
        marginal_graph = self.marginalize(nodes)
        return nx.is_directed_acyclic_graph(marginal_graph.graph)

    def get_permuted_graph(self) -> ADMG:
        node_permutation = {n: n_new for (n, n_new) in zip(self.graph.nodes, np.random.permutation(self.graph.nodes))}
        return ADMG(nx.relabel_nodes(self.graph, node_permutation, copy=True))

    @staticmethod
    def _marginalise_node(graph: nx.DiGraph, node: Any, confounded_nodes: Set) -> Tuple[nx.DiGraph, Set[Any]]:
        # Add egdes from all predecessors to sucessors, i.e. replace  x -> node -> y with a new edge x -> y
        for pre in graph.predecessors(node):
            for succ in graph.successors(node):
                if pre != succ:
                    graph.add_edge(pre, succ)
        for suc_one in graph.successors(node):  # Add bidericted edge if node is confounder of nodes
            for suc_two in graph.successors(node):
                if suc_two != suc_one:
                    confounded_nodes.add((suc_one, suc_two))
                    confounded_nodes.add((suc_two, suc_one))

            # Add (x, suc_one) to confounded nodes if z is removed, z -> suc_one and (x, z) are confounded, i.e.
            # if they are 'indirectly' confounded
            for x, _ in [(x, y) for (x, y) in confounded_nodes if y == node]:
                confounded_nodes.add((x, suc_one))
                confounded_nodes.add((suc_one, x))

        graph.remove_node(node)
        return graph, confounded_nodes

    @staticmethod
    def load_graph(graph_dir: str) -> ADMG:
        return ADMG(nx.read_graphml(graph_dir))

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
                    dot_edge = pydot.Edge(node_one, node_two, arrowtail='none', arrowhead='none',
                                          style='dotted')
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
            ground_truth_admg = ADMG(ground_truth)
            marginalised_ground_truth = ground_truth_admg.marginalize(self.variables())
            return super().eval_all_metrics(marginalised_ground_truth.graph)
        else:
            return super().eval_all_metrics(ground_truth)
