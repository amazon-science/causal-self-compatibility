# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import math
from typing import Callable, List, Any

import numpy as np
import numpy.linalg
import pandas as pd
from causal_graphs.sc_causal_graph import SCCausalGraph
from roca import RoCA


class SelfCompatibilityScorer:
    def __init__(self, algorithm: Callable[[pd.DataFrame], SCCausalGraph], num_subsets: int, subset_size: float = .5,
                 score_type: str = 'graphical', restrict_num_points: int = None):
        self.algorithm = algorithm
        self.num_subsets = num_subsets
        self.subset_size = subset_size
        self.score_type = score_type
        self.restrict_num_points = restrict_num_points
        self.marginal_graphs = []

    def _draw_subsets(self, graph: SCCausalGraph) -> List[List[Any]]:
        variables = graph.variables()
        subset_size = math.floor(len(variables) * self.subset_size)
        subset_list = []
        for _ in range(self.num_subsets):
            subset = np.random.choice(list(variables), size=subset_size, replace=False)
            subset_list.append(subset)
        return subset_list

    def _compatibility(self, joint_graph: SCCausalGraph, marginal_graphs: List[SCCausalGraph],
                       data: pd.DataFrame = None) -> float:
        if self.score_type == 'graphical':
            return SelfCompatibilityScorer._graphical_compatibility(joint_graph, marginal_graphs)
        elif self.score_type == 'interventional':
            return SelfCompatibilityScorer._interventional_compatibility(joint_graph, marginal_graphs, data)
        else:
            raise ValueError()

    @staticmethod
    def _graphical_compatibility(joint_graph: SCCausalGraph, marginal_graphs: List[SCCausalGraph]) -> float:
        shds = []
        for marginal_graph in marginal_graphs:
            marginalised_joint_graph = joint_graph.marginalize(marginal_graph.variables())
            shds.append(marginal_graph.shd(marginalised_joint_graph))
        return float(np.mean(shds))

    @staticmethod
    def _interventional_compatibility(joint_graph: SCCausalGraph, marginal_graphs: List[SCCausalGraph],
                                      data: pd.DataFrame) -> float:
        test_results = []
        for x, y in [(x, y) for x in joint_graph.variables() for y in joint_graph.variables() if x != y]:
            # Add adjustment set from joint graph
            adj_set = joint_graph.get_adjustment_set(x, y)
            if joint_graph.adjustment_valid(adj_set, x, y):
                # the adjustment sets are stored as frozensets, so they can be added to a set
                # this way we prevent to have the same adjustment set mutliple times
                adjustment_sets = {frozenset(adj_set)}
            else:
                adjustment_sets = set([])
            # If x and y are in a certain subset, add the adjustment set from the respective marginal graph
            for marginal_graph in marginal_graphs:
                subset = marginal_graph.variables()
                if x in subset and y in subset:
                    adj_set = marginal_graph.get_adjustment_set(x, y)
                    if marginal_graph.adjustment_valid(adj_set, x, y):
                        adjustment_sets.add(frozenset(adj_set))

            if len(adjustment_sets) > 1:
                try:
                    incompatibility = not RoCA().test_causal_strength_identical(data, x, y, adjustment_sets)
                    test_results.append(1 if incompatibility else 0)
                except numpy.linalg.LinAlgError as err:
                    test_results.append(1)  # Count as incompatibility
                    logging.error(err)
                    logging.error(x)
                    logging.error(y)
                    logging.error(adjustment_sets)
            else:
                marginal_ajd_incompatible = []
                for marginal_graph in marginal_graphs:
                    if x in marginal_graph.variables() and y in marginal_graph.variables():
                        joint_graph_marginalised = joint_graph.marginalize(marginal_graph.variables())
                        joint_adj = joint_graph_marginalised.get_adjustment_set(x, y)
                        joint_adj_valid = joint_graph_marginalised.adjustment_valid(joint_adj, x, y)
                        marginal_adj = marginal_graph.get_adjustment_set(x, y)
                        marginal_adj_valid = marginal_graph.adjustment_valid(marginal_adj, x, y)
                        marginal_ajd_incompatible.append(joint_adj_valid != marginal_adj_valid)
                if np.any(marginal_ajd_incompatible):
                    # If none are incompatible, we add nothing, as we neither an incompatibility nor a compatibility
                    test_results.append(1)
        return float(np.mean(test_results)) if test_results else 0.

    def compatibility_score(self, data: pd.DataFrame, joint_graph: SCCausalGraph = None) -> float:
        if self.restrict_num_points is not None:
            data = data.iloc[:self.restrict_num_points, :]

        if joint_graph is None:
            joint_graph = self.algorithm(data)

        self.marginal_graphs = []
        subset_list = self._draw_subsets(joint_graph)
        for subs in subset_list:
            self.marginal_graphs.append(self.algorithm(data[list(subs)]))

        return self._compatibility(joint_graph, self.marginal_graphs, data)
