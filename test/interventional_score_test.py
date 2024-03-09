# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest

import networkx as nx
import numpy as np
import pandas as pd

from src.causal_graphs.admg import ADMG
from src.self_compatibility import SelfCompatibilityScorer


class RCDTest(unittest.TestCase):

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

        est_joint = nx.DiGraph()
        est_joint.add_edge('x', 'z')
        est_joint.add_edge('y', 'z')
        est_joint = ADMG(est_joint)

        est_margial = nx.DiGraph()
        est_margial.add_edge('x', 'z')
        est_margial = ADMG(est_margial)

        score = SelfCompatibilityScorer._interventional_compatibility(joint_graph=est_joint,
                                                                      marginal_graphs=[est_margial],
                                                                      data=df)
        self.assertEqual(0, score)

        est_margial = nx.DiGraph()
        est_margial.add_edge('z', 'x')
        est_margial = ADMG(est_margial)

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
        x = np.random.normal(size=n_samples)
        y = alpha*x + np.random.normal(size=n_samples)
        nz = np.random.normal(size=n_samples)
        z = beta * y + nz
        v = gamma * x + delta * z + np.random.normal(size=n_samples)
        data = np.zeros((n_samples, 4))
        data[:, 0] = x
        data[:, 1] = y
        data[:, 2] = z
        data[:, 3] = v
        df = pd.DataFrame(data, columns=['x', 'y', 'z', 'v'])

        est_joint = nx.DiGraph()
        est_joint.add_edge('x', 'y')
        est_joint.add_edge('y', 'z')
        est_joint.add_edge('z', 'v')
        est_joint.add_edge('x', 'v')
        est_joint = ADMG(est_joint)

        est_margial_one = nx.DiGraph()
        est_margial_one.add_edge('x', 'z')
        est_margial_one.add_edge('z', 'v')
        est_margial_one.add_edge('x', 'v')
        est_margial_one = ADMG(est_margial_one)

        est_margial_two = nx.DiGraph()
        est_margial_two.add_edge('x', 'y')
        est_margial_two.add_edge('y', 'v')
        est_margial_two.add_edge('x', 'v')
        est_margial_two = ADMG(est_margial_two)

        score = SelfCompatibilityScorer._interventional_compatibility(joint_graph=est_joint,
                                                                      marginal_graphs=[est_margial_one, est_margial_two],
                                                                      data=df)
        self.assertEqual(0, score)

        est_margial_one = nx.DiGraph()
        est_margial_one.add_edge('x', 'z')
        est_margial_one.add_edge('z', 'v')
        est_margial_one = ADMG(est_margial_one)
        score = SelfCompatibilityScorer._interventional_compatibility(joint_graph=est_joint,
                                                                      marginal_graphs=[est_margial_one, est_margial_two],
                                                                      data=df)
        self.assertNotEqual(0, score)

if __name__ == '__main__':
    unittest.main()
