import argparse
import json
import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from pingouin import partial_corr

from src.cache_source_files import copy_referenced_files_to

parser = argparse.ArgumentParser(description='Generate plots that visualize self-compatibility and metrics.')
parser.add_argument('--comparison_metric', default='shd', help='metric by which to evaluate models. SHD by default.')
parser.add_argument('--comparison_metric_label', default='SHD', help='Name of metric to show up on axis.')
parser.add_argument('--no_title', action='store_true', help='Whether to show the figure title.')

params = vars(parser.parse_args())

# log file content
copy_referenced_files_to(__file__, "src_dump/")

# make font larger
font = {'weight': 'bold', 'size': 16}
plt.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# load data
metrics = pd.read_csv('../metrics.csv', index_col=0)
compatibility_metrics = pd.read_csv('self_compatibility.csv', index_col=0)
metrics = metrics.join(compatibility_metrics, rsuffix='_compatibility')
# which algo was used?
with open('../params.json', 'r') as file:
    discovery_params = json.load(file)
    algo = discovery_params['algo']
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

x_name = params['comparison_metric']
y_name = 'self_compatibility'
controll_name = 'avg_gt_degree'
# calculated partial correlation given the density of the ground truth graph.
# we do this is we assume this to be a confounder between SHD and Kappa.
corr = partial_corr(data=metrics, x=x_name, y=y_name, covar=controll_name)

# Scatter plot SHD vs Kappa
if not params["no_title"]:
    plt.title('{}: partial correlation={:.2f}, p={:.0e}'.format(algo, corr['r'].iloc[0], corr['p-val'].iloc[0]))
plt.scatter(metrics[x_name], metrics[y_name])
plt.xlabel(params['comparison_metric_label'])
plt.ylabel(r"Incompatibility score " + score_name)
plt.tight_layout()
plt.savefig("plot_{}_controlled.pdf".format(params['comparison_metric']))

plt.clf()

# Histogram of scores
if not params["no_title"]:
    plt.title('{}: Histogram'.format(algo))
plt.hist(metrics['self_compatibility'])
plt.xlabel(r"Incompatibility score " + score_name)
plt.ylabel("Number of datasets")
plt.tight_layout()
plt.savefig("plot_sb_hist.pdf")
