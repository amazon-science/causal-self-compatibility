# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import logging
import random
import shutil
import time
from pathlib import Path

import cdt
import networkx as nx
import numpy as np
from generator import DataGenerator


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Generate folder with data file.')
    subparsers = parser.add_subparsers(dest='command')
    synthetic_parser = subparsers.add_parser('synthetic', help='generate synthetic data')
    synthetic_parser.add_argument('--seed', default=112, type=int)
    synthetic_parser.add_argument('--num_nodes', default=10, type=int)
    synthetic_parser.add_argument('--num_points', default=1000, type=int)
    synthetic_parser.add_argument('--num_graphs', default=100, type=int)
    synthetic_parser.add_argument('--min_abs_coeff', default=.1, type=float)
    synthetic_parser.add_argument('--max_abs_coeff', default=1, type=float)
    synthetic_parser.add_argument('--noise', default="gaussian")
    synthetic_parser.add_argument('--mechanism', default="linear")
    synthetic_parser.add_argument('--noise_ceoff', default=1, type=float)
    synthetic_parser.add_argument('--expected_degree', default=2, type=int)
    synthetic_parser.add_argument('--num_hidden', default=0, type=int)
    subparsers.add_parser('sachs')
    params = vars(parser.parse_args())

    data_path = '../data/benchmark_' + time.strftime('%y.%m.%d_%H.%M.%S/data/')
    Path(data_path).mkdir(parents=True, exist_ok=False)
    src_dump = data_path + "src_dump/"
    Path(src_dump).mkdir(parents=True, exist_ok=True)
    shutil.copy('generate_data.py', src_dump + "generate_data.py")
    shutil.copy('generator.py', src_dump + "generator.py")

    with open(data_path + 'params.json', 'w') as file:
        json.dump(params, file)

    if params['command'] == 'sachs':
        i = 0
        data, graph = cdt.data.load_dataset('sachs')
        graph.remove_edge('PIP2', 'PIP3')       # Fix error in cdt. Cf. the original paper
        graph.add_edge('PIP3', 'PIP2')
        # Relabel feature as it causes trouble with the statsmodels format strings in the RoCA module
        data["pff_ft"] = data["p44/42"]
        data = data.drop("p44/42", axis=1)
        graph = nx.relabel_nodes(graph, {n: "pff_ft" if n == "p44/42" else n for n in graph.nodes})

        data.to_csv(data_path + 'dataset{}.csv'.format(i))
        nx.write_graphml(graph, data_path + 'graph{}.gml'.format(i))

    elif params['command'] == 'synthetic':

        seed = params['seed']
        np.random.seed(seed)
        random.seed(seed)

        for i in range(params['num_graphs']):
            logging.info('Generated Dataset {}'.format(i))

            num_vars_overall = params['num_nodes'] + params['num_hidden']
            g = DataGenerator(num_nodes=num_vars_overall, mechanism=params['mechanism'], graph_type='dag',
                              erdos_p=params['expected_degree'] / (num_vars_overall - 1),
                              coeff_range=(params['min_abs_coeff'], params['max_abs_coeff']))
            data, ground_truth = g.generate(num_samples=params['num_points'], noise=params['noise'],
                                            var=params['noise_ceoff'])

            observable_keys = np.random.choice(data.keys(), params['num_nodes'], replace=False)
            data = data.loc[:, observable_keys]
            data.to_csv(data_path + 'dataset{}.csv'.format(i))
            nx.write_graphml(ground_truth, data_path + 'graph{}.gml'.format(i))

    else:
        raise ValueError(params['command'])
