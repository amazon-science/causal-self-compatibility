# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest

import networkx as nx
import numpy as np

from src.causal_graphs.admg import ADMG
from src.m_separated import m_separated


class ADMGTest(unittest.TestCase):

    def test_marginalize_returns_expected_graph_for_m_graph(self):
        dag = nx.DiGraph()
        dag.add_edge('a', 'x')
        dag.add_edge('a', 'b')
        dag.add_edge('c', 'b')
        dag.add_edge('c', 'y')

        subset = {'x', 'y'}
        marginal_admg = ADMG(dag).marginalize(subset)
        self.assertTrue(np.all(subset == set(marginal_admg.variables())))
        self.assertTrue(self.all_seps_agree(marginal_admg.graph, dag))

        expected_dag = nx.DiGraph()
        expected_dag.add_nodes_from(['x', 'y'])
        self.assertEqual(0, marginal_admg.shd(expected_dag))

        expected_dag.add_edge('x', 'y')
        self.assertNotEqual(0, marginal_admg.shd(expected_dag))

    def test_marginalize_returns_expected_graph_for_chain_graph(self):
        dag = nx.DiGraph()
        dag.add_edge('a', 'b')
        dag.add_edge('b', 'c')
        dag.add_edge('c', 'd')

        subset = {'a', 'c'}
        marginal_admg = ADMG(dag).marginalize(subset)
        self.assertTrue(np.all(subset == set(marginal_admg.variables())))
        self.assertTrue(self.all_seps_agree(marginal_admg.graph, dag))

        expected_dag = nx.DiGraph()
        expected_dag.add_edge('a', 'c')
        self.assertEqual(0, marginal_admg.shd(expected_dag))

        expected_dag.remove_edge('a', 'c')
        self.assertNotEqual(0, marginal_admg.shd(expected_dag))

    def test_marginalize_returns_expected_graph_for_unshielded_collider_graph(self):
        dag = nx.DiGraph()
        dag.add_edge('a', 'b')
        dag.add_edge('c', 'b')
        dag.add_edge('d', 'c')

        subset = {'a', 'c'}
        marginal_admg = ADMG(dag).marginalize(subset)
        self.assertTrue(np.all(subset == set(marginal_admg.variables())))
        self.assertTrue(self.all_seps_agree(marginal_admg.graph, dag))

        expected_dag = nx.DiGraph()
        expected_dag.add_nodes_from(['a', 'c'])
        self.assertEqual(0, marginal_admg.shd(expected_dag))

        expected_dag.add_edge('a', 'c')
        self.assertNotEqual(0, marginal_admg.shd(expected_dag))

    def test_marginalize_returns_expected_graph_for_pure_confounding_graph(self):
        dag = nx.DiGraph()
        dag.add_edge('a', 'b')
        dag.add_edge('a', 'c')

        subset = {'b', 'c'}
        marginal_admg = ADMG(dag).marginalize(subset)
        self.assertTrue(np.all(subset == set(marginal_admg.variables())))
        self.assertTrue(self.all_seps_agree(marginal_admg.graph, dag))

        expected_dag = nx.DiGraph()
        expected_dag.add_nodes_from(['b', 'c'])
        expected_dag.add_edge('b', 'c')
        expected_dag.add_edge('c', 'b')
        self.assertEqual(0, marginal_admg.shd(expected_dag))

        expected_dag.remove_edge('b', 'c')
        self.assertNotEqual(0, marginal_admg.shd(expected_dag))

    def test_marginalize_returns_expected_graph_for_chain_of_bidirected_edges(self):
        admg = nx.DiGraph()
        admg.add_edge('a', 'b')
        admg.add_edge('b', 'a')
        admg.add_edge('b', 'c')
        admg.add_edge('c', 'b')

        subset = {'a', 'c'}
        marginal_admg = ADMG(admg).marginalize(subset)
        self.assertTrue(np.all(subset == set(marginal_admg.variables())))
        self.assertTrue(self.all_seps_agree(marginal_admg.graph, admg))

        expected_dag = nx.DiGraph()
        expected_dag.add_nodes_from(['a', 'c'])
        expected_dag.add_edge('a', 'c')
        expected_dag.add_edge('c', 'a')
        self.assertEqual(0, marginal_admg.shd(expected_dag))

        expected_dag.remove_edge('a', 'c')
        self.assertNotEqual(0, marginal_admg.shd(expected_dag))

    def test_marginalize_returns_expected_graph_for_indirect_confounding(self):
        admg = nx.DiGraph()
        admg.add_edge('a', 'b')
        admg.add_edge('b', 'a')
        admg.add_edge('b', 'c')

        subset = {'a', 'c'}
        marginal_admg = ADMG(admg).marginalize(subset)
        self.assertTrue(np.all(subset == set(marginal_admg.variables())))
        self.assertTrue(self.all_seps_agree(marginal_admg.graph, admg))

        expected_dag = nx.DiGraph()
        expected_dag.add_nodes_from(['a', 'c'])
        expected_dag.add_edge('a', 'c')
        expected_dag.add_edge('c', 'a')
        self.assertEqual(0, marginal_admg.shd(expected_dag))

        expected_dag.remove_edge('a', 'c')
        self.assertNotEqual(0, marginal_admg.shd(expected_dag))

    @staticmethod
    def all_seps_agree(dag_one: nx.DiGraph, dag_two: nx.DiGraph):
        seps_agree = []
        for x, y in [(x, y) for x in dag_one.nodes for y in dag_one.nodes if x != y]:
            for z in [] + [z for z in dag_one.nodes if z != x and z != y]:
                original_sep = m_separated(dag_one, x, y, set(z))
                sep_after = m_separated(dag_two, x, y, set(z))
                seps_agree.append(original_sep == sep_after)
        return all(seps_agree)


if __name__ == '__main__':
    unittest.main()
