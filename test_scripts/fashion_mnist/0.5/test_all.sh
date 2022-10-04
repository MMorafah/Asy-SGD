#!/bin/bash
#SBATCH --time=40:00:00

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --partition=cpulong
#SBATCH --job-name=asysonatampi2nodemnist
#SBATCH --err=myJob.err
#SBATCH --out=myJob.out

ml TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4
ml matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
ml SciPy-bundle/2019.10-fosscuda-2019b-Python-3.7.4

python ../../../src/test_all_nodes.py --iter 55000 --dataset fashion_mnist --connectivity 0.5 --snapshot 30 --seed 0 --len_shards 200 --save_path ../../../save_async/ 2>&1 | tee ../../../save_async/fashion_mnist/0.5/test_all_logs.txt