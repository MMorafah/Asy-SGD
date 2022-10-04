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

python ../../../src/test_agent.py --num_nodes 24 --iter 65000 --model resnet_18 --dataset cifar --batch_size 200 --connectivity 0.9 --snapshot 30 --seed 0 --len_shards 200 --save_path ../../../save_async/ 2>&1 | tee ../../../save_async/cifar/0.9/24/test_logs.txt