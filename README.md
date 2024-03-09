# Code for "Self-Compatibility"

## Install
Install requirements via
        
    pip -r requirements.txt

To generate the SID results you also need an installation of `R` with the packages `SID` and  `readr`.

## Running experiments
**See respective files for command line options of every command**

### Data and Experiments
Data can be generated via 

    python generate_data synthetic 
    python generate_data sachs 

Where the latter formats the Sachs dataset such that it can be used by the rest of the code.

The directories of the datasets have a time stamp. To run the experiments, type e.g.

    cd ../../data/benchmark_23.09.26_14.05.39
    python ../../src/causal_discovery.py --algo FCI

and for the self-compatibility test e.g.

    python ../../src/self_benchmark.py --algo FCI

For the results with the structural interventional distance, run

    Rscript calc_SID.R

after the actual causal discovery in the directory of the form `benchmark_*/ALGO_*` and then the respective plot commands.

### Plots

To generate the correlation plots as in Fig. 2 and the example graphs enter the (time stamped) directory of the self-compatibility test, so e.g.

    cd RCD_23.09.26_14.06.52/self_benchmark_23.09.26_14.38.36
    python ../../../../src/plot_correlation.py
    python ../../../../src/plot_example_graphs.py

To generate the model selection plots, you have to be in the directory of the benchmark, i.e. `benchmark_*` and the run

    python ../../../../src/plot_model_selection.py parameters

or 

    python ../../../../src/plot_model_selection.py algorithms

after generating the respective results via `causal_discovery.py` and `self_benchmark.py`.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.


