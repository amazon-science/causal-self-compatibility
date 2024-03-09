# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest

import networkx as nx
import numpy as np
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph

from src.causal_graphs.pag import PAG


class MarginalisationTest(unittest.TestCase):

    def test_d_sep_single_collider(self):
        dag = nx.DiGraph()
        dag.add_edge('x', 'z')
        dag.add_edge('y', 'z')

        pag = PAG(PAG._dag_to_pag(dag))
        mag = pag.draw_random_mag()
        dag_after = self._cl_dag_to_nx(PAG._mag_to_canonic_dag(mag)[0])
        self.assertTrue(self.all_seps_agree(dag, dag_after))

    def test_d_sep_confounder(self):
        dag = nx.DiGraph()
        dag.add_edge('z', 'x')
        dag.add_edge('z', 'y')

        pag = PAG(PAG._dag_to_pag(dag))
        mag = pag.draw_random_mag()
        dag_after = self._cl_dag_to_nx(PAG._mag_to_canonic_dag(mag)[0])
        self.assertTrue(self.all_seps_agree(dag, dag_after))

    def test_d_sep_complete_dag(self):
        dag = nx.DiGraph()
        dag.add_edge('z', 'x')
        dag.add_edge('x', 'y')
        dag.add_edge('z', 'y')

        pag = PAG(PAG._dag_to_pag(dag))
        mag = pag.draw_random_mag()
        dag_after = self._cl_dag_to_nx(PAG._mag_to_canonic_dag(mag)[0])
        self.assertTrue(self.all_seps_agree(dag, dag_after))

    def test_d_sep_m_graph(self):
        dag = nx.DiGraph()
        dag.add_edge('a', 'x')
        dag.add_edge('a', 'b')
        dag.add_edge('c', 'b')
        dag.add_edge('c', 'y')

        pag = PAG(PAG._dag_to_pag(dag))
        mag = pag.draw_random_mag()
        dag_after = self._cl_dag_to_nx(PAG._mag_to_canonic_dag(mag)[0])
        self.assertTrue(self.all_seps_agree(dag, dag_after))

    def test_marginalisation_m_graph(self):
        dag = nx.DiGraph()
        dag.add_edge('a', 'x')
        dag.add_edge('a', 'b')
        dag.add_edge('c', 'b')
        dag.add_edge('c', 'y')

        pag = PAG(PAG._dag_to_pag(dag))
        subset = {'x', 'y'}
        marginal_pag = pag.marginalize(subset)
        self.assertTrue(np.all(subset == set(marginal_pag.variables())))

        expected_dag = nx.DiGraph()
        expected_dag.add_nodes_from(['x', 'y'])
        self.assertEqual(0, marginal_pag.shd(expected_dag))

        expected_dag.add_edge('x', 'y')
        self.assertNotEqual(0, marginal_pag.shd(expected_dag))

    def test_marginalisation_chain(self):
        dag = nx.DiGraph()
        dag.add_edge('a', 'b')
        dag.add_edge('b', 'c')
        dag.add_edge('c', 'd')

        pag = PAG(PAG._dag_to_pag(dag))
        subset = {'a', 'c'}
        marginal_pag = pag.marginalize(subset)
        self.assertTrue(np.all(subset == set(marginal_pag.variables())))

        expected_dag = nx.DiGraph()
        expected_dag.add_edge('a', 'c')
        self.assertEqual(0, marginal_pag.shd(expected_dag))

        expected_dag.remove_edge('a', 'c')
        self.assertNotEqual(0, marginal_pag.shd(expected_dag))

    def test_marginalisation_collider(self):
        dag = nx.DiGraph()
        dag.add_edge('a', 'b')
        dag.add_edge('c', 'b')
        dag.add_edge('d', 'c')

        pag = PAG(PAG._dag_to_pag(dag))
        subset = {'a', 'c'}
        marginal_pag = pag.marginalize(subset)
        self.assertTrue(np.all(subset == set(marginal_pag.variables())))

        expected_dag = nx.DiGraph()
        expected_dag.add_nodes_from(['a', 'c'])
        self.assertEqual(0, marginal_pag.shd(expected_dag))

        expected_dag.add_edge('a', 'c')
        self.assertNotEqual(0, marginal_pag.shd(expected_dag))

    @staticmethod
    def all_seps_agree(dag_one: nx.DiGraph, dag_two: nx.DiGraph):
        seps_agree = []
        for x, y in [(x, y) for x in dag_one.nodes for y in dag_one.nodes if x != y]:
            for z in [] + [z for z in dag_one.nodes if z != x and z != y]:
                original_sep = nx.d_separated(dag_one, {x}, {y}, set(z))
                sep_after = nx.d_separated(dag_two, {x}, {y}, set(z))
                seps_agree.append(original_sep == sep_after)
        return all(seps_agree)


    @staticmethod
    def _cl_dag_to_nx(graph: GeneralGraph) -> nx.DiGraph:
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
                raise ValueError('Did not expect edge with two tails')
        return g_hat


if __name__ == '__main__':
    unittest.main()
