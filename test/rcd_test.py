# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest

import networkx as nx
import numpy as np
import pandas as pd

from src.algo_wrappers import RCDLINGAM


class RCDTest(unittest.TestCase):

    def test_single_collider(self):
        n_samples = 3000
        alpha = 2 * np.random.rand()
        beta = 2 * np.random.rand()
        x = np.random.uniform(-1, 1, size=n_samples)
        y = np.random.uniform(-1, 1, size=n_samples)
        nz = np.random.uniform(-1, 1, size=n_samples)
        z = alpha * x + beta * y + nz
        data = np.zeros((n_samples, 3))
        data[:, 0] = x
        data[:, 1] = y
        data[:, 2] = z
        df = pd.DataFrame(data, columns=['x', 'y', 'z'])
        gt = nx.DiGraph()
        gt.add_edge('x', 'z')
        gt.add_edge('y', 'z')
        admg = RCDLINGAM(alpha=0.01)(data=df)
        self.assertEqual(len(admg.graph.edges), 2)
        self.assertTrue(admg.graph.has_edge('x', 'z'))
        self.assertTrue(admg.graph.has_edge('y', 'z'))
        self.assertEqual(admg.shd(gt), 0)

    def test_complete_dag(self):
        n_samples = 3000
        alpha = 2 * np.random.rand()
        beta = 2 * np.random.rand()
        gamma = 2 * np.random.rand()
        n_x = np.random.uniform(-1, 1, size=n_samples)
        n_y = np.random.uniform(-1, 1, size=n_samples)
        n_z = np.random.uniform(-1, 1, size=n_samples)
        z = n_z
        x = beta * z + n_x
        y = alpha * x + gamma * z + n_y
        data = np.zeros((n_samples, 3))
        data[:, 0] = x
        data[:, 1] = y
        data[:, 2] = z
        df = pd.DataFrame(data, columns=['x', 'y', 'z'])
        gt = nx.DiGraph()
        gt.add_edge('z', 'x')
        gt.add_edge('x', 'y')
        gt.add_edge('z', 'y')
        admg = RCDLINGAM(alpha=0.01)(data=df)
        self.assertEqual(len(admg.graph.edges), 3)
        self.assertEqual(admg.shd(gt), 0)

    def test_confounder(self):
        n_samples = 3000
        beta = 2 * np.random.rand()
        gamma = 2 * np.random.rand()
        noisex = np.random.uniform(-1, 1, size=n_samples)
        ny = np.random.uniform(-1, 1, size=n_samples)
        nz = np.random.uniform(-1, 1, size=n_samples)
        z = nz
        x = beta * z + noisex
        y = gamma * z + ny
        data = np.zeros((n_samples, 3))
        data[:, 0] = x
        data[:, 1] = y
        data[:, 2] = z
        df = pd.DataFrame(data, columns=['x', 'y', 'z'])
        gt = nx.DiGraph()
        gt.add_edge('z', 'x')
        gt.add_edge('z', 'y')
        admg = RCDLINGAM(alpha=0.01)(data=df)
        self.assertEqual(len(admg.graph.edges), 2)
        self.assertEqual(admg.shd(gt), 0)


if __name__ == '__main__':
    unittest.main()
