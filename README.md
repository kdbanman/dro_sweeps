# dro_sweeps

For common Distributionally Robust Optimization methods, study hyperparameter effects on synthetic fairness problems.

## Setup

Create a virtual environment, activate it, update it, and install dependencies

```
virtualenv env
ln -s env/bin/activate activate
source activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

# TODO

## New Investment

- create `results = run(params)` type function
- save results to csv
- make runner which runs `run` if results not exist
- sweep in detail with overzealous training steps
- results in postgres

## Bugs

- why don't we learn the right classifier?

## Debt
- change `outputs` to `labels` in places dro.py and upstream places.
- add plotting source file from classification example notebook functions.
- make imports consistent across all files.
- Support `weights` that are just tuples, rather than jnp.arrays, so that the functions are responsible for shape, etc.
