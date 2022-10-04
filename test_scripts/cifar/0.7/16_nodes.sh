#!/bin/bash
#SBATCH --time=70:00:00

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH --partition=cpulong
#SBATCH --job-name=asysonatampi2nodemnist
#SBATCH --err=TEST_cifar10_16nodes.err
#SBATCH --out=TEST_cifar10_16nodes.out

ml TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4
ml matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
ml SciPy-bundle/2019.10-fosscuda-2019b-Python-3.7.4

dir='../../../save_async_step_reduce/cifar/0.7/16'

python ../../../src/test_agent.py --num_nodes 16 \
--iter 30000 \
--model resnet_18 \
--dataset cifar \
--batch_size 200 \
--connectivity 0.7 \
--snapshot 30 \
--seed 0 \
--len_shards 200 \
--save_path ../../../save_async_step_reduce/ \
2>&1 | tee $dir'/test_logs.txt'
