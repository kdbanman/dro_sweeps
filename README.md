# dro_sweeps

For common Distributionally Robust Optimization methods, study hyperparameter effects on synthetic fairness problems.

## Setup

On a Compute Canada SLURM cluster, create a virtual environment and activate it

```
module load python/3.9
virtualenv env
ln -s env/bin/activate activate
source activate
```

On Niagara, install the dependencies via the pip index, because JAX is confusingly not available otherwise:

```
pip install -r requirements.txt
```

On other clusters, install the dependencies via local modules:

```
pip install -r requirements.txt --no-index
```

## Development

Activate the virtual environment

```
source activate
```

## Run

```
sbatch sbatch_run.sh
```
