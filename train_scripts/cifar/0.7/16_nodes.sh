#!/bin/bash
#SBATCH --time=70:00:00

#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --partition=cpulong
#SBATCH --job-name=asysonatampi16nodecifar
#SBATCH --err=TRAIN_cifar10_16nodes.err
#SBATCH --out=TRAIN_cifar10_16nodes.out

ml TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4
ml matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
ml SciPy-bundle/2019.10-fosscuda-2019b-Python-3.7.4

dir='../../../save_async_step_reduce/cifar/0.7/16'
if [ ! -e $dir ]; then
mkdir -p $dir
fi

mpiexec -np 16 python ../../../src/main.py --iter 30000 \
--step 0.5 \
--eps 0.0001 \
--model resnet_18 \
--dataset cifar \
--batch_size 128 \
--connectivity 0.7 \
--snapshot 30 \
--len_shards 200 \
--seed 0 \
--gpu 12 \
--lr_schedule step_reduce \
--lr_reduce_rate 2 \
--lr_iter_reduce 10000 \
--save_path ../../../save_async_step_reduce/ \
2>&1 | tee $dir'/train_logs.txt'
