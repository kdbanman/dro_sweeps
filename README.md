# dro_sweeps

For common Distributionally Robust Optimization methods, study hyperparameter effects on synthetic fairness problems.

## Setup

On a Compute Canada SLURM cluster, create a virtual environment and activate it

```
module load python/3.9
virtualenv env
source env/bin/activate
```

On Niagara, install the dependencies via the pip index, because JAX is confusingly not available otherwise:

```
pip install -r requirements.txt
```

On other clusters, install the dependencies via local modules:

```
./install_dependencies_no_internet.sh
```

## Development

Load the relevant module and activate the virtual environment:

```
source cc_setup.sh
```

## Run

```
sbatch sbatch_run.sh
```
