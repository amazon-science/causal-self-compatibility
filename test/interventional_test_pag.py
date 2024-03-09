# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest

import networkx as nx
import numpy as np
import pandas as pd
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode

from src.causal_graphs.pag import PAG
from src.self_compatibility import SelfCompatibilityScorer


class InterventionalPAGTest(unittest.TestCase):

    def test_single_collider(self):
        n_samples = 300
        alpha = 2 * np.random.rand()
        beta = 2 * np.random.rand()
        x = np.random.normal(size=n_samples)
        y = np.random.normal(size=n_samples)
        nz = np.random.normal(size=n_samples)
        z = alpha * x + beta * y + nz
        data = np.zeros((n_samples, 3))
        data[:, 0] = x
        data[:, 1] = y
        data[:, 2] = z
        df = pd.DataFrame(data, columns=['x', 'y', 'z'])

        est_joint = GeneralGraph([GraphNode("x"), GraphNode("y"), GraphNode("z")])
        est_joint.add_edge(Edge(est_joint.get_node("x"), est_joint.get_node("z"), Endpoint.CIRCLE, Endpoint.ARROW))
        est_joint.add_edge(Edge(est_joint.get_node("y"), est_joint.get_node("z"), Endpoint.CIRCLE, Endpoint.ARROW))
        est_joint = PAG(est_joint)

        est_margial = GeneralGraph([GraphNode("x"), GraphNode("z")])
        est_margial.add_edge(Edge(est_margial.get_node("x"), est_margial.get_node("z"), Endpoint.CIRCLE, Endpoint.ARROW))
        est_margial = PAG(est_margial)

        score = SelfCompatibilityScorer._interventional_compatibility(joint_graph=est_joint,
                                                                      marginal_graphs=[est_margial],
                                                                      data=df)
        self.assertEqual(0, score)

        est_margial = GeneralGraph([GraphNode("x"), GraphNode("z")])
        est_margial.add_edge(
            Edge(est_margial.get_node("z"), est_margial.get_node("x"), Endpoint.CIRCLE, Endpoint.ARROW))
        est_margial = PAG(est_margial)

        score = SelfCompatibilityScorer._interventional_compatibility(joint_graph=est_joint,
                                                                      marginal_graphs=[est_margial],
                                                                      data=df)
        self.assertNotEqual(0, score)

    def test_confounder(self):
        n_samples = 600
        alpha = 2 * np.random.rand()
        beta = 2 * np.random.rand()
        gamma = 2 * np.random.rand()
        delta = 2 * np.random.rand()
        epsilon = 2 * np.random.rand()
        zeta = 2 * np.random.rand()
        a = np.random.normal(size=n_samples)
        b = np.random.normal(size=n_samples)
        x = np.random.normal(size=n_samples)
        y = alpha*x + np.random.normal(size=n_samples)
        nz = np.random.normal(size=n_samples)
        z = beta * y + epsilon * a + nz
        v = gamma * x + delta * z * zeta * b + np.random.normal(size=n_samples)
        data = np.zeros((n_samples, 6))
        data[:, 0] = x
        data[:, 1] = y
        data[:, 2] = z
        data[:, 3] = v
        data[:, 4] = a
        data[:, 5] = b
        df = pd.DataFrame(data, columns=['x', 'y', 'z', 'v', 'a', 'b'])

        est_joint = nx.DiGraph()
        est_joint.add_edge('x', 'y')
        est_joint.add_edge('y', 'z')
        est_joint.add_edge('z', 'v')
        est_joint.add_edge('x', 'v')
        est_joint.add_edge('a', 'z')
        est_joint.add_edge('b', 'v')
        est_joint = PAG(PAG._dag_to_pag(est_joint))

        est_margial_one = nx.DiGraph()
        est_margial_one.add_edge('x', 'z')
        est_margial_one.add_edge('z', 'v')
        est_margial_one.add_edge('x', 'v')
        est_margial_one.add_edge('a', 'z')
        est_margial_one.add_edge('b', 'v')
        est_margial_one = PAG(PAG._dag_to_pag(est_margial_one))

        est_margial_two = nx.DiGraph()
        est_margial_two.add_edge('x', 'y')
        est_margial_two.add_edge('y', 'v')
        est_margial_two.add_edge('x', 'v')
        est_margial_two.add_edge('b', 'v')
        est_margial_two = PAG(PAG._dag_to_pag(est_margial_two))

        score = SelfCompatibilityScorer._interventional_compatibility(joint_graph=est_joint,
                                                                      marginal_graphs=[est_margial_one, est_margial_two],
                                                                      data=df)
        self.assertLess(score, .1)

        est_margial_one = nx.DiGraph()
        est_margial_one.add_edge('x', 'z')
        est_margial_one.add_edge('v', 'z')
        est_margial_one.add_edge('a', 'z')
        est_margial_one.add_edge('b', 'v')
        est_margial_one = PAG(PAG._dag_to_pag(est_margial_one))
        score = SelfCompatibilityScorer._interventional_compatibility(joint_graph=est_joint,
                                                                      marginal_graphs=[est_margial_one, est_margial_two],
                                                                      data=df)
        self.assertNotEqual(0, score)

if __name__ == '__main__':
    unittest.main()
