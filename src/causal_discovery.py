# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import glob
import json
import logging
import random
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from algo_wrappers import PC, GES, RCDLINGAM, FCI
from cache_source_files import copy_referenced_files_to

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Parse command line args
    parser = argparse.ArgumentParser(description='Do causal discovery on the dataset in the current folder.')
    parser.add_argument('--seed', default=11, type=int)
    parser.add_argument('--algo', default="PC")
    parser.add_argument('--alpha', default=None, type=float)
    parser.add_argument('--lambda', default=None, type=float)
    parser.add_argument('--restrict_num_points', default=None, type=int)
    parser.add_argument('--dir', default='.')
    parser.add_argument('--indep_test', default='fisherz', type=str)
    params = vars(parser.parse_args())

    # Seed
    np.random.seed(params['seed'])
    random.seed(params['seed'])

    curr_dir = params['dir']
    # load params that were used for data generation
    with open(curr_dir + '/data/params.json', 'r') as file:
        data_params = json.load(file)

    # create folder for data and copy src files there
    result_path = curr_dir + '/{}_'.format(params['algo']) + time.strftime('%y.%m.%d_%H.%M.%S/')
    Path(result_path).mkdir(parents=True, exist_ok=False)
    graph_dir = result_path + 'graphs/'
    Path(graph_dir).mkdir(parents=True, exist_ok=False)

    copy_referenced_files_to(__file__, result_path + "src_dump/")

    # Init algo with correct parameters
    if params['algo'] == 'PC':
        if params['alpha'] is None:
            params['alpha'] = 0.01
        algo = PC(params['alpha'], params['indep_test'])
    elif params['algo'] == 'GES':
        if params['lambda'] is None:
            params['lambda'] = 0.01
        algo = GES(params['lambda'])
    elif params['algo'] == 'RCD':
        if params['alpha'] is None:
            params['alpha'] = 0.01
        algo = RCDLINGAM(params['alpha'])
    elif params['algo'] == 'FCI':
        if params['alpha'] is None:
            params['alpha'] = 0.01
        algo = FCI(params['alpha'], params['indep_test'])
    else:
        raise NotImplementedError()

    # Store cmd parameters
    with open(result_path + 'params.json', 'w') as file:
        json.dump(params, file)

    metrics_results = []
    for i in range(len(glob.glob(curr_dir + '/data/dataset*.csv'))):
        logging.info('Dataset {}'.format(i))
        data = pd.read_csv(curr_dir + '/data/dataset{}.csv'.format(i), index_col=0)
        # If parameter is set, drop some datapoints
        if params['restrict_num_points']:
            data = data.iloc[:params['restrict_num_points'], :]
        # Call algorithm
        g_hat = algo(data)
        # Store graph
        g_hat.save_graph(graph_dir + 'g_hat{}.gml'.format(i))
        # Calculate all evaluation metrics
        ground_truth = nx.read_graphml(curr_dir + '/data/graph{}.gml'.format(i))
        metrics_results.append(g_hat.eval_all_metrics(ground_truth))

    # Store the evaluation metrics in a csv file
    metric_frame = pd.DataFrame(metrics_results)
    metric_frame.to_csv(result_path + 'metrics.csv')
