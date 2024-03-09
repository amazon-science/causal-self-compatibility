# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd


class DataGenerator:

    def __init__(self, num_nodes: int = 8, erdos_p: float = None, mechanism: str = 'linear', graph_type: str = 'dag',
                 coeff_range: Tuple[float, float] = (0.1, 1)):
        if graph_type != 'dag':
            raise NotImplementedError('Graph Type can only be "dag" right now. Not ' + str(graph_type))
        self.num_nodes = num_nodes
        self.nodes = ['V{}'.format(i+1) for i in range(num_nodes)]
        self.p = (1.1 * np.log(num_nodes)) / num_nodes if erdos_p is None else erdos_p
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.nodes)
        self.causal_order = np.random.permutation(self.nodes)
        self.parent_idxs = defaultdict(lambda:[])
        df_idx = {n:i for (i, n) in enumerate(self.nodes)}
        for i, x in enumerate(self.causal_order):
            for j, y in enumerate(self.causal_order):
                if j > i and np.random.rand() < self.p:
                    self.parent_idxs[y].append(df_idx[x])
                    self.graph.add_edge(x, y)

        self.mechanism = {}
        for node in self.parent_idxs.keys():
            if mechanism == 'linear':
                self.mechanism[node] = LinearMechanism(len(self.parent_idxs[node]), coeff_range)
            else:
                raise NotImplementedError('Non-linear model not implemented yet')

    def generate(self, num_samples: int = 100, var: float = 1, noise: str = 'gaussian') -> Tuple[
        pd.DataFrame, nx.DiGraph]:
        sample = pd.DataFrame(np.zeros((num_samples, self.num_nodes)), columns=self.nodes)
        if noise == 'gaussian':
            n_func = lambda: np.random.normal(loc=0, scale=var)
        elif noise == 'uniform':
            a = np.sqrt(3 * var)  # get var as variance
            n_func = lambda: np.random.uniform(low=-a, high=a)
        else:
            raise NotImplementedError('Invalid noise parameter: {}'.format(noise))
        for k in range(num_samples):
            for node in self.causal_order:
                value = n_func()  # Right now only additive noise
                if node in self.mechanism:
                    value += self.mechanism[node](sample.iloc[k, self.parent_idxs[node]])
                sample.loc[k, node] = value
        return sample, self.graph


class LinearMechanism:
    def __init__(self, num_parents: int, coeff_range: Tuple[float, float], weights: np.array=None):
        if weights is None:
            self.weights = np.random.uniform(low=coeff_range[0], high=coeff_range[1], size=num_parents) \
                           * np.random.choice([-1, 1])
        else:
            self.weights = weights

    def __call__(self, parents: np.array) -> np.array:
        return np.dot(self.weights, parents)
