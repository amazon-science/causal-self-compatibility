import argparse
import glob
import json
import matplotlib
import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from src.cache_source_files import copy_referenced_files_to

parser = argparse.ArgumentParser(description='Generate plots that visualize model selection with self-compatibility.')
subparsers = parser.add_subparsers(dest='command')
parser.add_argument('--comparison_metric', default='shd', help='metric by which to evaluate models. SHD by default.')
parser.add_argument('--comparison_metric_label', default='SHD', help='Name of metric to show up on axis.')
parser.add_argument('--no_title', action='store_true', help='Whether to show the figure title.')
algorithm_parser = subparsers.add_parser('algorithms', help='model selection between algorithms')
algorithm_parser.add_argument('--first_algo', default='PC', type=str)
algorithm_parser.add_argument('--second_algo', default='GES', type=str)
parameter_parser = subparsers.add_parser('parameters', help='model selection between parameters of one algorithm')
parameter_parser.add_argument('--algo', default='PC', type=str)
parameter_parser.add_argument('--parameter', default='alpha', type=str)
params = vars(parser.parse_args())

copy_referenced_files_to(__file__, "data/src_dump/")

font = {'weight': 'bold', 'size': 16}
plt.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

if params['command'] == 'algorithms':
    relevant_directories = [np.sort(glob.glob('{}*'.format(alg)))[-1] for alg in
                            [params['first_algo'], params['second_algo']]]
elif params['command'] == 'parameters':
    relevant_directories = np.sort(glob.glob('{}*'.format(params['algo'])))
else:
    raise NotImplementedError()

compatibility_key = 'self_compatibility'  # Which column of the result dataframe to use
metrics = []
for algo_dir in relevant_directories:
    # load file with metrics of causal discovery result
    met = pd.read_csv(algo_dir + '/metrics.csv')
    # load results of self compatibility and add column of interest to the metrics dataframe
    sb_dir = np.sort(glob.glob(algo_dir + '/self_benchmark*'))[-2]
    met[compatibility_key] = pd.read_csv(sb_dir + '/self_compatibility.csv')[compatibility_key]
    met[compatibility_key] = met[compatibility_key].apply(lambda s: 0 if np.isnan(s) else s)
    # load parameters of causal discovery
    with open(algo_dir + '/params.json', 'r') as file:
        # which algo was used?
        discovery_params = json.load(file)
        met['algo'] = discovery_params['algo']
        # if we select between parameter settings, also add the parameter to the dataframe
        if params['command'] == 'parameters' and discovery_params[params['parameter']] is not None:
            met[params['parameter']] = discovery_params[params['parameter']]
    metrics.append(met)
    # which compatibility score was used? Only used for axis labels. Assumes that all folders contain the same kappa.
    with open(sb_dir + '/params.json', 'r') as file:
        compatibility_params = json.load(file)
        score_type = compatibility_params['compatibility']
        if score_type == 'interventional':
            score_name = r'$\kappa^I$'
        elif score_type == 'graphical':
            score_name = r'$\kappa^G$'
        else:
            raise NotImplementedError()

result_frame = pd.concat(metrics)

# Add column that indicates the rank (w.r.t. self compatibility) of the row among all rows that refer to the same
# dataset
# As the individual datasets of each algorithm run are concatenated, the index contains the index of the dataset
# I.e. grouping by index groups together rows that correspond to the same dataset
result_frame['rank'] = result_frame.groupby(result_frame.index)[compatibility_key].rank('first')
# Needed to compare frames
result_frame.sort_index(inplace=True)
# Determine winners as the best run (w.r.t. to the rank from above) and losers as the worst
winners = result_frame[result_frame['rank'] == 1]
losers = result_frame[result_frame['rank'] == len(relevant_directories)]

# Plot histograms
_, ax = plt.subplots()
t, p = scipy.stats.ttest_ind(winners[params['comparison_metric']], losers[params['comparison_metric']])
if not params["no_title"]:
    ax.set_title('t={:.2f}, p={:.2f}'.format(t, p))
ax.hist(winners[params['comparison_metric']], alpha=.5, label='Winners', density=True)
ax.hist(losers[params['comparison_metric']], alpha=.5, label='Losers', density=True)
ax.set_xlabel(params['comparison_metric_label'])
ax.legend()
plt.savefig("plot_model_selection_hist_{}.pdf".format(params['comparison_metric']))

plt.close()


def size_adjusted_scatter(x: pd.Series, y: pd.Series):
    _, ax = plt.subplots()
    # Count frequency of points to plot them in different sizes
    freq = np.zeros_like(x)
    for i in range(len(winners)):
        freq[i] = np.sum(np.logical_and(x == x.iloc[i], y == y.iloc[i]))

    # Define the size of each dot based on the frequency
    dot_scale_factor = 25.  # magic number so the dots look reasonable in the plot
    dot_size = freq * dot_scale_factor
    # Sizes to show in the legend
    legend_sizes = [np.min(dot_size), np.median(dot_size), np.quantile(dot_size, .75)]
    # Create scatter plot. Plot for every dot size separately, to be able to decide which one to add to legend (a bit hacky)
    num_datasets_legend = []
    for d in np.unique(dot_size):
        idx = dot_size == d  # Create index mask for points that have dot_size d
        if d in legend_sizes:
            ax.scatter(x[idx], y[idx], s=dot_size[idx], # Fix colour, so it won't rotate for next d
                       # Also rescale dot size to actual number of points in label
                       label="{} datasets".format(int(d / 25.)), c='C0'
                       )
            num_datasets_legend.append(int(d / 25.))
        else:
            ax.scatter(x[idx], y[idx], s=dot_size[idx], c='C0')
    return ax, np.max(num_datasets_legend)


# Scatter plot
ax, max_dot_size = size_adjusted_scatter(winners[params['comparison_metric']], losers[params['comparison_metric']])
# Plot diagonal line
ma = max(result_frame[params['comparison_metric']])
mi = min(result_frame[params['comparison_metric']])
ax.plot([mi, ma], [mi, ma])

ax.set_xlabel('{} of Winner'.format(params['comparison_metric_label']))
ax.set_ylabel('{}  of Loser'.format(params['comparison_metric_label']))

precentage_correct = np.mean(winners[params['comparison_metric']] <= losers[params['comparison_metric']])
percentage_strictly_above = np.mean(winners[params['comparison_metric']] < losers[params['comparison_metric']])
percentage_strictly_below = np.mean(winners[params['comparison_metric']] > losers[params['comparison_metric']])
percentage_on_line = np.mean(winners[params['comparison_metric']] == losers[params['comparison_metric']])
print('strictly above: ', percentage_strictly_above)
print('strictly below: ', percentage_strictly_below)
print('on line: ', percentage_on_line)
if not params["no_title"]:
    plt.title('Model Selection: {:.0f}% correct decisions'.format(precentage_correct * 100))
if max_dot_size > 1:
    plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('plot_model_selection_scatter_{}.pdf'.format(params['comparison_metric']))

plt.close()

# Difference in kappa vs difference in shd
comp_diff = losers[compatibility_key] - winners[compatibility_key]
metric_diff = losers[params['comparison_metric']] - winners[params['comparison_metric']]
ax, max_dot_size = size_adjusted_scatter(comp_diff, metric_diff)
ax.plot([comp_diff.min(), comp_diff.max()], [0, 0])
ax.set_xlabel(r'Difference in {}'.format(score_name))
ax.set_ylabel('Difference in {}'.format(params['comparison_metric_label']))
if not params["no_title"]:
    plt.title('Differences in compatibility and {}'.format(params['comparison_metric_label']))
plt.tight_layout()
if max_dot_size > 1:
    plt.legend()
plt.savefig('plot_model_selection_differences_{}.pdf'.format(params['comparison_metric']))
