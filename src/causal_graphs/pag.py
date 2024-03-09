from __future__ import annotations

import io
import logging
from typing import Tuple, List, Iterable, Any, Union, Dict, Set

import networkx as nx
import numpy as np
from causallearn.graph.Dag import Dag
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Node import Node
from causallearn.utils.DAG2PAG import dag2pag
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
from matplotlib import image as mpimg

from src.causal_graphs.sc_causal_graph import SCCausalGraph
from src.m_separated import m_separated
from src.metrics import skeleton_tpr, skeleton_fpr, skeleton_precision, skeleton_f1, avg_gt_degree


class PAG(SCCausalGraph):

    def __init__(self, graph: GeneralGraph):
        super().__init__(graph)

    def skeleton_tpr(self, ground_truth: nx.DiGraph) -> float:
        # skeleton of PAG, MAG and DAG are the same.
        # for ease of implementation we convert the PAG to a (networkx) MAG and reuse the skeleton score for
        # networkx DiGraphs
        g_hat_mag = self.draw_random_mag()
        return skeleton_tpr(ground_truth, g_hat_mag)

    def skeleton_fpr(self, ground_truth: nx.DiGraph) -> float:
        # skeleton of PAG, MAG and DAG are the same.
        # for ease of implementation we convert the PAG to a (networkx) MAG and reuse the skeleton score for
        # networkx DiGraphs
        g_hat_mag = self.draw_random_mag()
        return skeleton_fpr(ground_truth, g_hat_mag)

    def skeleton_precision(self, ground_truth: nx.DiGraph) -> float:
        # skeleton of PAG, MAG and DAG are the same.
        # for ease of implementation we convert the PAG to a (networkx) MAG and reuse the skeleton score for
        # networkx DiGraphs
        g_hat_mag = self.draw_random_mag()
        return skeleton_precision(ground_truth, g_hat_mag)

    def skeleton_f1(self, ground_truth: nx.DiGraph) -> float:
        # skeleton of PAG, MAG and DAG are the same.
        # for ease of implementation we convert the PAG to a (networkx) MAG and reuse the skeleton score for
        # networkx DiGraphs
        g_hat_mag = self.draw_random_mag()
        return skeleton_f1(ground_truth, g_hat_mag)

    def avg_gt_degree(self, ground_truth: nx.DiGraph) -> float:
        return avg_gt_degree(ground_truth, self.graph)  # TODO should this compare to the ground truth pag?

    def avg_degree(self, _: nx.DiGraph) -> float:
        return float(np.mean([self.graph.get_degree(node) for node in self.graph.get_nodes()]))

    def shd(self, ground_truth: Union[nx.DiGraph, SCCausalGraph]) -> int:
        if type(ground_truth) == nx.DiGraph:
            ground_truth = self._dag_to_pag(ground_truth)
        elif type(ground_truth) == PAG:
            ground_truth = ground_truth.graph
        else:
            raise NotImplementedError()
        errors = 0
        for x, y in [(x.get_name(), y.get_name()) for i, x in enumerate(ground_truth.get_nodes()) for j, y in
                     enumerate(ground_truth.get_nodes()) if i < j]:
            if ground_truth.is_adjacent_to(ground_truth.get_node(x), ground_truth.get_node(y)):
                gt_edge = ground_truth.get_edge(ground_truth.get_node(x), ground_truth.get_node(y))
                if not self.graph.is_adjacent_to(self.graph.get_node(x), self.graph.get_node(y)):
                    errors += 1
                    errors += 1 if gt_edge.get_endpoint1() != Endpoint.CIRCLE else 0
                    errors += 1 if gt_edge.get_endpoint2() != Endpoint.CIRCLE else 0
                else:
                    hat_edge = self.graph.get_edge(self.graph.get_node(x), self.graph.get_node(y))
                    if gt_edge.get_endpoint1() != hat_edge.get_endpoint1():
                        errors += 1
                    if gt_edge.get_endpoint2() != hat_edge.get_endpoint2():
                        errors += 1
            else:
                if self.graph.is_adjacent_to(self.graph.get_node(x), self.graph.get_node(y)):
                    hat_edge = self.graph.get_edge(self.graph.get_node(x), self.graph.get_node(y))
                    errors += 1
                    errors += 1 if hat_edge.get_endpoint1() != Endpoint.CIRCLE else 0
                    errors += 1 if hat_edge.get_endpoint2() != Endpoint.CIRCLE else 0
        return errors

    @staticmethod
    def valid_metrics() -> List[str]:
        return ['shd', 'skeleton_f1', 'avg_degree', 'avg_gt_degree', 'skeleton_tpr', 'skeleton_fpr',
                'skeleton_precision']

    def variables(self) -> List[Any]:
        return [n.get_name() for n in self.graph.get_nodes()]

    def marginalize(self, remaining_nodes: Iterable[Any]) -> PAG:
        mag = self.draw_random_mag()
        canonic_dag, dummy_latents = self._mag_to_canonic_dag(mag)
        latent_nodes = [canonic_dag.get_node(n) for n in
                        set(self.variables()).difference(remaining_nodes)] + dummy_latents
        marginalised_pag = dag2pag(canonic_dag, islatent=latent_nodes)
        return PAG(marginalised_pag)

    def save_graph(self, graph_dir: str):
        with open(graph_dir, 'w') as file:
            file.write(str(self.graph))

    def draw_random_mag(self) -> nx.DiGraph:
        aag = nx.DiGraph()
        ccgraph = nx.DiGraph()
        aag.add_nodes_from([n.get_name() for n in self.graph.get_nodes()])
        for edge in self.graph.get_graph_edges():
            if edge.get_endpoint1() == Endpoint.CIRCLE and edge.get_endpoint2() == Endpoint.ARROW:
                aag.add_edge(edge.get_node1().get_name(), edge.get_node2().get_name())  # o-> becomes ->
            elif edge.get_endpoint2() == Endpoint.CIRCLE and edge.get_endpoint1() == Endpoint.ARROW:
                aag.add_edge(edge.get_node2().get_name(), edge.get_node1().get_name())  # <-o becomes <-
            elif edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.ARROW:
                aag.add_edge(edge.get_node1().get_name(), edge.get_node2().get_name())  # keep directed edges
            elif edge.get_endpoint2() == Endpoint.TAIL and edge.get_endpoint1() == Endpoint.ARROW:
                aag.add_edge(edge.get_node2().get_name(), edge.get_node1().get_name())
            elif edge.get_endpoint1() == Endpoint.ARROW and edge.get_endpoint2() == Endpoint.ARROW:
                aag.add_edge(edge.get_node1().get_name(), edge.get_node2().get_name())
                aag.add_edge(edge.get_node2().get_name(), edge.get_node1().get_name())
            else:
                ccgraph.add_edge(edge.get_node1().get_name(), edge.get_node2().get_name())
                ccgraph.add_edge(edge.get_node2().get_name(), edge.get_node1().get_name())

        mag = aag
        while len(ccgraph.nodes) >= 1:
            old_num_nodes = len(ccgraph.nodes)
            # remove "disconnected nodes", i.e. nodes with no undirected edge and no outgoing directed edge left
            ccgraph.remove_nodes_from([n for n in ccgraph.nodes if ccgraph.out_degree(n) == 0])
            remaining_nodes = list(ccgraph.nodes)
            for x in remaining_nodes:
                # if node is sink
                directed_outgoing = [y for y in ccgraph.successors(x) if not ccgraph.has_edge(y, x)]
                if len(directed_outgoing) == 0:
                    # If orientation creates only shielded colliders
                    all_neighbours = {y for y in list(ccgraph.successors(x)) + list(ccgraph.predecessors(x))}
                    bidirected_neighbours = [y for y in ccgraph.successors(x) if ccgraph.has_edge(y, x)]
                    only_shielded = all(
                        [all([ccgraph.has_edge(z, y) or ccgraph.has_edge(y, z) for z in all_neighbours if z != y]) for y
                         in bidirected_neighbours])
                    if only_shielded:
                        # Orient all bidirected edges towards x in mag
                        for y in bidirected_neighbours:
                            mag.add_edge(y, x)
                        # Remove n from ccgraph
                        ccgraph.remove_node(x)
            if old_num_nodes == len(ccgraph.nodes):
                logging.warning('PAG is no equivalence class of MAGs!')
                # Orient the first x in the remaining graph
                x = remaining_nodes[0]
                for y in [y for y in ccgraph.successors(x) if ccgraph.has_edge(y, x)]:
                    mag.add_edge(y, x)
                # Remove n from ccgraph
                ccgraph.remove_node(x)
        return mag

    @staticmethod
    def _mag_to_canonic_dag(mag: nx.DiGraph) -> Tuple[Dag, List[Node]]:
        # Replace bidirected edges by dummy latent node
        bidirected_edges = set([])
        for x, y in mag.edges:
            if mag.has_edge(y, x) and (y, x) not in bidirected_edges:
                bidirected_edges.add((x, y))
        nodes = [GraphNode(name) for name in mag.nodes] + [GraphNode('L{}'.format(i)) for i in
                                                           range(len(bidirected_edges))]
        dag = Dag(nodes)
        dag_nx = nx.DiGraph()
        dag_nx.add_nodes_from([n.get_name() for n in nodes])
        for x, y in mag.edges:
            if not mag.has_edge(y, x):
                dag.add_directed_edge(dag.get_node(x), dag.get_node(y))
                dag_nx.add_edge(x, y)
        for i, (x, y) in enumerate(bidirected_edges):
            dag.add_directed_edge(dag.get_node('L{}'.format(i)), dag.get_node(x))
            dag.add_directed_edge(dag.get_node('L{}'.format(i)), dag.get_node(y))
            dag_nx.add_edge('L{}'.format(i), x)
            dag_nx.add_edge('L{}'.format(i), y)

        while not nx.is_directed_acyclic_graph(dag_nx):  # If due to finite sample effects there is a circle, remove it
            logging.warning('g_hat is no proper PAG! Contains cycle.')
            cycle = nx.find_cycle(dag_nx, orientation='original')
            edge = cycle[0]
            dag_nx.remove_edge(edge[0], edge[1])
            dag.remove_connecting_edge(dag.get_node(edge[0]), dag.get_node(edge[1]))
        return dag, [dag.get_node('L{}'.format(i)) for i in range(len(bidirected_edges))]

    @staticmethod
    def _dag_to_pag(graph: nx.DiGraph) -> GeneralGraph:
        if not nx.is_directed_acyclic_graph(graph):
            logging.warning('Ground truth is not a DAG!')
            graph = graph.copy()
            while not nx.is_directed_acyclic_graph(
                    graph):  # If due to finite sample effects there is a circle, remove it
                cycle = nx.find_cycle(graph, orientation='original')
                edge = cycle[0]
                graph.remove_edge(edge[0], edge[1])
        nodes_dict = {name: GraphNode(name) for name in graph.nodes}
        dag = Dag(list(nodes_dict.values()))
        for (src, trgt) in graph.edges:
            dag.add_directed_edge(dag.get_node(src), dag.get_node(trgt))
        pag = dag2pag(dag, islatent=[])
        return pag

    def get_adjustment_set(self, x: Any, y: Any) -> Set[Any]:
        # Return the canonical adjustment set from Def 12 in Perkovic et al.
        # (except for zero effects. see comment below)
        x_node = self.graph.get_node(x)
        y_node = self.graph.get_node(y)
        forb = self._collect_possible_descendants(x_node)
        # In the description in the paper, Y is removed from the set that's called "parents" here.
        # We only subtract {Y} from the possible ancestors of Y (and not from possible ancestors of X), because of the
        # hacky way we deal with anticausal directions. If we test the anticausal direction, Y will be in the adjustment
        # set. Then the regression in the test of Su & Henckel will give a zero as coefficient for X (see other files).
        parents = self._collect_possible_ancestors(x_node).union(self._collect_possible_ancestors(y_node) - {y_node})
        canonical_adj_set = {n.get_name() for n in parents - forb.union({x_node})}
        # If Y is not a possible descendant of X, we test, if all marginal graphs have a coefficient of zero.
        # The canonical adjustment set is usually big, so that we get many false positives in the Su & Henckel test.
        # On the contrary, when we always return the empty set in this case, we almost always conduct no test at all,
        # as we don't get more than one distinct conditioning set. Therefore, we return a random singleton set.
        if y not in forb and canonical_adj_set:
            if y in parents:
                return {y}
            else:
                return {canonical_adj_set.pop()}
        return canonical_adj_set

    def adjustment_valid(self, adjustment_set: Set[Any], x: Any, y: Any) -> bool:
        # Implementation of the generalised adjustment criterion https://jmlr.org/papers/volume18/16-319/16-319.pdf
        x_node = self.graph.get_node(x)
        if self.is_amenable(x, y):
            forb = set(self._collect_possible_descendants(x_node))
            if not adjustment_set.intersection(forb):
                return not self._exists_prop_def_status_ncp(adjustment_set, x, y)
        return False

    def _exists_prop_def_status_ncp(self, adjustment_set: Set[Any], x: Any, y: Any) -> bool:
        x_node = self.graph.get_node(x)
        backdoor_graph = GeneralGraph([])
        backdoor_graph.transfer_nodes_and_edges(self.graph)
        neighbours = self.graph.get_adjacent_nodes(x_node)
        for neighbour in neighbours:
            edge = self.graph.get_edge(x_node, neighbour)
            if edge.get_proximal_endpoint(x_node) != Endpoint.ARROW:
                backdoor_graph.remove_edge(edge)

        backdoor_mag = PAG(backdoor_graph).draw_random_mag()
        return not m_separated(backdoor_mag, x, y, adjustment_set)

    def is_amenable(self, x: Any, y: Any):
        x_node = self.graph.get_node(x)
        y_node = self.graph.get_node(y)
        for node in self.graph.get_adjacent_nodes(x_node):
            edge = self.graph.get_edge(x_node, node)
            if (edge.get_proximal_endpoint(x_node) != Endpoint.ARROW
                    and self._exists_possibly_directed_path(node, y_node, {x_node})):
                if edge.get_proximal_endpoint(x_node) != Endpoint.TAIL or not self.is_edge_visible(x, node.get_name()):
                    return False
        return True

    def is_edge_visible(self, x: Any, y: Any) -> bool:
        x_node = self.graph.get_node(x)
        y_node = self.graph.get_node(y)
        if self.graph.get_directed_edge(x_node, y_node) is None:
            raise ValueError("No definitely directed edge between {} and {}".format(x, y))
        for pa in self.graph.get_adjacent_nodes(x_node):
            edge = self.graph.get_edge(pa, x_node)
            if edge.get_distal_endpoint(pa) == Endpoint.ARROW and self.graph.get_edge(pa, y_node) is None:
                return True
        for pa in self.graph.get_parents(y_node):
            if self._collider_path_in_parents(x_node, y_node, pa, {x_node}):
                return True
        return False

    def _collider_path_in_parents(self, x: Node, y: Node, prev_node: Node, visited: Set[Node]) -> bool:
        for c in self.graph.get_parents(y):
            if c not in visited:
                edge = self.graph.get_edge(c, x)
                if self.graph.is_def_collider(prev_node, c, x) and edge.get_distal_endpoint(c) == Endpoint.ARROW:
                    return True
                for next_node in self.graph.get_parents(y):
                    if (self.graph.is_def_collider(prev_node, c, next_node)
                            and self._collider_path_in_parents(x, y, c, visited.union({c}))):
                        return True
        return False

    def _collect_possible_descendants(self, node: Node, descendants: Set[Node] = None) -> Set[Node]:
        remove_starting_node_later = False
        if descendants is None:
            descendants = {node}
            remove_starting_node_later = True
        elif node in descendants:
            return descendants
        descendants = descendants.union({node})
        neighbours = self.graph.get_adjacent_nodes(node)
        for neighbour in neighbours:
            edge = self.graph.get_edge(node, neighbour)
            if edge.get_proximal_endpoint(node) != Endpoint.ARROW:
                descendants = descendants.union(self._collect_possible_descendants(neighbour, descendants))
        return descendants.difference({node}) if remove_starting_node_later else descendants

    def _collect_possible_ancestors(self, node: Node, ancestors: Set[Node] = None) -> Set[Node]:
        if ancestors is None:
            ancestors = {node}
        elif node in ancestors:
            return ancestors
        ancestors = ancestors.union({node})
        neighbours = self.graph.get_adjacent_nodes(node)
        for neighbour in neighbours:
            edge = self.graph.get_edge(node, neighbour)
            proximal_endpoint = edge.get_proximal_endpoint(node)
            if ((proximal_endpoint == Endpoint.ARROW or proximal_endpoint == Endpoint.CIRCLE)
                    and edge.get_distal_endpoint(node) != Endpoint.ARROW):
                ancestors = ancestors.union(self._collect_possible_ancestors(neighbour, ancestors))
        return ancestors

    def _exists_possibly_directed_path(self, x: Node, y: Node, visited: Set[Node] = None) -> bool:
        if visited is None:
            visited = {x}
        elif x in visited:
            return False
        if x == y:
            return True
        neighbours = self.graph.get_adjacent_nodes(x)
        for neighbour in neighbours:
            edge = self.graph.get_edge(x, neighbour)
            if edge.get_proximal_endpoint(x) != Endpoint.ARROW:
                if self._exists_possibly_directed_path(neighbour, y, visited.union({x})):
                    return True
        return False

    @staticmethod
    def load_graph(graph_dir: str) -> PAG:
        return PAG(txt2generalgraph(graph_dir))

    def visualize(self, ax=None):
        pyd = GraphUtils.to_pydot(self.graph, labels=self.variables())
        tmp_png = pyd.create_png(f="png")
        fp = io.BytesIO(tmp_png)
        img = mpimg.imread(fp, format='png')
        ax.set_axis_off()
        ax.imshow(img)

    def eval_all_metrics(self, ground_truth: nx.DiGraph) -> Dict[str, float]:
        if len(ground_truth.nodes) > len(self.variables()):
            var_set = set(self.variables())
            ground_truth.remove_nodes_from([node for node in ground_truth.nodes if node not in var_set])
        return super().eval_all_metrics(ground_truth)
