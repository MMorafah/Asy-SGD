#!/bin/bash
#SBATCH --time=70:00:00

#SBATCH --nodes=8
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=1
#SBATCH --mem=70G
#SBATCH --partition=cpulong
#SBATCH --job-name=asysonatampi24nodemnist_II
#SBATCH --err=test_24nodes_II.err
#SBATCH --out=test_24nodes_II.out

ml TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4
ml matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
ml SciPy-bundle/2019.10-fosscuda-2019b-Python-3.7.4

python ../../../src/test_agent.py --num_nodes 24 --iter 55000 --model cnn --dataset mnist --batch_size 200 --connectivity 0.7 --snapshot 30 --seed 0 --len_shards 200 --save_path ../../../save_async_II/ 2>&1 | tee ../../../save_async_II/mnist/0.7/24/test_logs.txt


