#!/bin/bash
#SBATCH -J test_run
#SBATCH --account=def-nidhih
#SBATCH --cpus-per-task=1
#SBATCH -t 0-00:10 # 10 min

source cc_setup.sh
source activate

python $HOME/proj/dro_sweeps/main.py
