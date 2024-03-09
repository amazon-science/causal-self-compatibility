# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import glob
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from algo_wrappers import PC, GES, RCDLINGAM, FCI
from cache_source_files import copy_referenced_files_to
from self_compatibility import SelfCompatibilityScorer

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Generate synthetic data.')
    parser.add_argument('--seed', default=11, type=int)
    parser.add_argument('--num_subsets', default=40, type=int)
    parser.add_argument('--subset_size', default=.5, type=float)
    parser.add_argument('--compatibility', default="graphical")
    parser.add_argument('--algo', default="PC")
    params = vars(parser.parse_args())

    # Set random seed
    seed = params['seed']
    np.random.seed(seed)
    random.seed(seed)
    # load parameters of data generation
    with open('data/params.json', 'r') as file:
        data_params = json.load(file)

    for cd_dir in glob.glob('{}*'.format(params['algo'])):
        # Create folder for results and store source code there
        result_path = cd_dir + '/self_benchmark_' + time.strftime('%y.%m.%d_%H.%M.%S/')
        Path(result_path).mkdir(parents=True, exist_ok=False)
        graph_dir = result_path + 'marginal_graphs/'
        Path(graph_dir).mkdir(parents=True, exist_ok=False)
        src_dump = result_path + "src_dump/"

        copy_referenced_files_to(__file__, result_path + "src_dump/")

        # Store command line parameters
        with open(result_path + 'params.json', 'w') as file:
            json.dump(params, file)
        # Load parameters from causal discovery
        with open(cd_dir + '/params.json', 'r') as file:
            discovery_params = json.load(file)

        # Init algo with correct parameters
        if params['algo'] == 'PC':
            algo = PC(discovery_params['alpha'], discovery_params['indep_test'])
        elif params['algo'] == 'GES':
            algo = GES(discovery_params['lambda'])
        elif params['algo'] == 'RCD':
            algo = RCDLINGAM(discovery_params['alpha'])
        elif params['algo'] == 'FCI':
            algo = FCI(discovery_params['alpha'], discovery_params['indep_test'])
        else:
            raise NotImplementedError()

        compatibility_list = []
        for i in range(len(glob.glob(cd_dir + "/graphs/g_hat*.gml"))):
            logging.info('Dataset {}'.format(i))
            data = pd.read_csv('data/dataset{}.csv'.format(i), index_col=0)
            # Load the graph with the respective method of the graph type that is returned by the algo
            joint_graph = algo.load_graph(cd_dir + '/graphs/g_hat{}.gml'.format(i))
            # Calculate the score according to the command line parameters
            scorer = SelfCompatibilityScorer(algo, params['num_subsets'], params['subset_size'],
                                             params['compatibility'], discovery_params['restrict_num_points'])

            compatibility_score = scorer.compatibility_score(data, joint_graph)
            graph_dir_i = graph_dir + 'dataset{}/'.format(i)
            Path(graph_dir_i).mkdir(parents=True, exist_ok=False)
            for j, marginal_graph in enumerate(scorer.marginal_graphs):
                marginal_graph.save_graph(graph_dir_i + 'marginal_graph{}.gml'.format(j))
            compatibility_list.append(compatibility_score)
        # Store the scores in a csv file
        data_cols = {'self_compatibility': compatibility_list}
        df = pd.DataFrame(data_cols)
        df.to_csv(result_path + 'self_compatibility.csv')
