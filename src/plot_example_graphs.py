import glob
import json
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from algo_wrappers import PC, GES, RCDLINGAM, FCI
from causal_graphs.admg import ADMG
from src.cache_source_files import copy_referenced_files_to

# log file content
copy_referenced_files_to(__file__, "src_dump/")

np.random.seed(43)

# make font larger
font = {'weight': 'bold', 'size': 16}
plt.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

N_ROWS = 3
N_SUBGRAPHS = 3

# load data
metrics = pd.read_csv('../metrics.csv', index_col=0)
compatibility_metrics = pd.read_csv('self_compatibility.csv', index_col=0)
metrics = metrics.join(compatibility_metrics, rsuffix='_compatibility')
# which algo was used?
with open('../params.json', 'r') as file:
    discovery_params = json.load(file)
    algo_name = discovery_params['algo']
# which compatibility score was used?
with open('params.json', 'r') as file:
    compatibility_params = json.load(file)
    score_type = compatibility_params['compatibility']
    if score_type == 'interventional':
        score_name = r'$\kappa^I$'
    elif score_type == 'graphical':
        score_name = r'$\kappa^G$'
    else:
        raise NotImplementedError()

# Init algo (params are not really necessary here)
if algo_name == 'PC':
    algo = PC(discovery_params['alpha'], discovery_params['indep_test'])
elif algo_name == 'GES':
    algo = GES(discovery_params['lambda'])
elif algo_name == 'RCD':
    algo = RCDLINGAM(discovery_params['alpha'])
elif algo_name == 'FCI':
    algo = FCI(discovery_params['alpha'], discovery_params['indep_test'])
else:
    raise NotImplementedError()

cd_dir = '..'
data_dir = '../../data'

fig = plt.figure(constrained_layout=True, figsize=(15, N_ROWS*3.33))
fig.suptitle('Example graphs for {}'.format(algo_name))
subfigs = fig.subfigures(N_ROWS, 1)
for subfi, i in zip(subfigs if N_ROWS > 1 else [subfigs],
                    np.random.choice(list(range(len(glob.glob(cd_dir + "/graphs/g_hat*.gml")))), size=N_ROWS)):
    ground_truth = ADMG.load_graph(data_dir + '/graph{}.gml'.format(i))
    # Load the graph with the respective method of the graph type that is returned by the algo
    joint_graph = algo.load_graph(cd_dir + '/graphs/g_hat{}.gml'.format(i))

    subfi.suptitle("Dataset Nr. {} with {} = {:.2f}".format(i, score_name, metrics['self_compatibility'].iloc[i]))
    axs = subfi.subplots(1, N_SUBGRAPHS + 2, width_ratios=[1.5, 1.5] + N_SUBGRAPHS*[1])
    ground_truth.visualize(axs[0])
    axs[0].set_title('Ground Truth')

    joint_graph.visualize(axs[1])
    axs[1].set_title('Joint')

    m_graph_dir = 'marginal_graphs/dataset{}'.format(i)
    for ax, j in zip(axs[2:], np.random.choice(list(range(len(glob.glob(m_graph_dir + '/marginal_graph*.gml')))),
                                                  size=N_SUBGRAPHS)):
        marginal_graph = algo.load_graph(m_graph_dir + '/marginal_graph{}.gml'.format(j))
        marginal_graph.visualize(ax)
        ax.set_title('Marginal {}'.format(j))

plt.savefig("plot_example_graphs.pdf")
plt.show()
plt.clf()
