import argparse
import json
import pandas as pd

from src.cache_source_files import copy_referenced_files_to

parser = argparse.ArgumentParser(description='Generate plots that visualize self-compatibility and metrics.')
parser.add_argument('--comparison_metric', default='shd', help='metric by which to evaluate models. SHD by default.')
parser.add_argument('--comparison_metric_label', default='SHD', help='Name of metric to show up on axis.')
params = vars(parser.parse_args())

# log file content
copy_referenced_files_to(__file__, "src_dump/")


# load data
metrics = pd.read_csv('../metrics.csv', index_col=0)
compatibility_metrics = pd.read_csv('self_compatibility.csv', index_col=0)
metrics = metrics.join(compatibility_metrics, rsuffix='_compatibility')
# which algo was used?
with open('../params.json', 'r') as file:
    discovery_params = json.load(file)
    algo = discovery_params['algo']
    indep = discovery_params['indep_test']
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

print(algo, indep)
print("{}: {:.2f}".format(score_name, metrics['self_compatibility'].iloc[0]))
print('SHD: {:.2f}'.format(metrics['shd'].iloc[0]))
print('F1: {:.2f}'.format(metrics['skeleton_f1'].iloc[0]))
