# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest

import cdt
import numpy as np
import pandas as pd
from cdt.data.causal_mechanisms import gaussian_cause

from src.causal_graphs.admg import ADMG
from src.causal_graphs.pag import PAG
from src.self_compatibility import SelfCompatibilityScorer


def random_algo(data: pd.DataFrame) -> ADMG:
    generator = cdt.data.AcyclicGraphGenerator('linear', 'uniform', nodes=6,
                                               npoints=4,
                                               noise_coeff=1,
                                               # It seems like there is a bug in cdt and the graph is too dense
                                               # if we don't divide by two
                                               expected_degree=2 / 2.,
                                               initial_variable_generator=gaussian_cause, dag_type='erdos')
    _, random_graph = generator.generate()
    return ADMG(random_graph).marginalize(data.keys())


def random_algo_pag(data: pd.DataFrame) -> PAG:
    generator = cdt.data.AcyclicGraphGenerator('linear', 'uniform', nodes=6,
                                               npoints=4,
                                               noise_coeff=1,
                                               # It seems like there is a bug in cdt and the graph is too dense
                                               # if we don't divide by two
                                               expected_degree=2 / 2.,
                                               initial_variable_generator=gaussian_cause, dag_type='erdos')
    _, random_graph = generator.generate()
    return PAG(PAG._dag_to_pag(random_graph)).marginalize(data.keys())


class IntegrationTest(unittest.TestCase):

    def test_admg_interventional(self):
        gt_results = []
        random_results = []
        for _ in range(3):
            generator = cdt.data.AcyclicGraphGenerator('linear', 'gaussian', nodes=6,
                                                       npoints=2000,
                                                       noise_coeff=1,
                                                       # It seems like there is a bug in cdt and the graph is too dense
                                                       # if we don't divide by two
                                                       expected_degree=2 / 2.,
                                                       initial_variable_generator=gaussian_cause, dag_type='erdos')
            data, ground_truth = generator.generate()
            gt_admg = ADMG(ground_truth)
            perfect_algo = lambda d: ADMG(ground_truth).marginalize(d.keys())
            scorer = SelfCompatibilityScorer(perfect_algo, 40, score_type='interventional')
            score_gt = scorer.compatibility_score(data, gt_admg)
            gt_results.append(score_gt)

            scorer = SelfCompatibilityScorer(random_algo, 40, score_type='interventional')
            score_random = scorer.compatibility_score(data)
            random_results.append(score_random)

        self.assertLessEqual(np.mean(gt_results), 0.3)
        self.assertGreater(np.mean(random_results), 0.6)

    def test_pag_interventional(self):
        gt_results = []
        random_results = []
        for _ in range(20):
            generator = cdt.data.AcyclicGraphGenerator('linear', 'gaussian', nodes=6,
                                                       npoints=3000,
                                                       noise_coeff=1,
                                                       # It seems like there is a bug in cdt and the graph is too dense
                                                       # if we don't divide by two
                                                       expected_degree=2 / 2.,
                                                       initial_variable_generator=gaussian_cause, dag_type='erdos')
            data, ground_truth = generator.generate()
            gt_pag = PAG(PAG._dag_to_pag(ground_truth))
            perfect_algo = lambda d: PAG(PAG._dag_to_pag(ground_truth)).marginalize(d.keys())
            scorer = SelfCompatibilityScorer(perfect_algo, 40, score_type='interventional')
            score_gt = scorer.compatibility_score(data, gt_pag)
            gt_results.append(score_gt)

            scorer = SelfCompatibilityScorer(random_algo_pag, 40, score_type='interventional')
            score_random = scorer.compatibility_score(data)
            random_results.append(score_random)

        self.assertLessEqual(np.mean(gt_results), 0.3)
        self.assertGreater(np.mean(random_results), 0.6)

if __name__ == '__main__':
    unittest.main()
