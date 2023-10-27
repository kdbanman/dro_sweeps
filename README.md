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

## Usage

The `dro_sweeps/` directory contains the important source code, including a performant minibatch implementation of CVaR DRO, as well as synthtetic fairness data generation.

The `classification_example.ipynb` and `regression_exmaple.py` files contain classification and regression examples using ^that source code.  From those, it should be clear how hyperparameter sweeps work for classification and regression.

# TODO

## Bugs

- why do we get NaN gradients with the current setup for `classification_example.ipynb`
    - it looks like the weights gradually grow until the losses and gradients explode.
    - I think this happens because (with an intercept) there are infinitely many ways to 
      parametrize the optimal separating hyperplane, and the problem somehow incentivizes larger weights.
    - Naively, this could be addressed by L2 regularization, but this would add an experimental variable
      which [significantly affects the question we are trying to address](https://arxiv.org/abs/1911.08731)
      not to mention it's another hyperparameter to sweep.
    - Instead, it might be better to not use an intercept, if it's indeed overparametrizing the problem.
    - remove debugging TODO comments once solved

## New Investment

- create `results = run(params)` type function
- create _objects_ or _dicts_ to encapsulate model, training config, training results
- make runner which runs `run` if results not exist
- sweep in detail with overzealous training steps
- results in postgres

## Debt

- change `outputs` to `labels` in places dro.py and upstream places.
- add plotting source file from classification example notebook functions.
- make imports consistent across all files.
- Support `weights` that are just tuples, rather than jnp.arrays, so that the functions are responsible for shape, etc.
