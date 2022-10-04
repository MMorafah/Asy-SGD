#!/bin/bash
#SBATCH --time=70:00:00

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --partition=cpulong
#SBATCH --job-name=asysonatampi4nodecifar
#SBATCH --err=myJob.err
#SBATCH --out=myJob.out

#DIR = "save_async/cifar"
#[ ! -d "$DIR" ] && mkdir -p "$DIR"

ml TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4
ml matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
ml SciPy-bundle/2019.10-fosscuda-2019b-Python-3.7.4

mpiexec -np 4 python ../../../src/main.py --iter 65000 --step 0.04 --eps 0.0001 --model resnet_18 --dataset cifar --batch_size 128 --connectivity 0.9 --snapshot 30 --len_shards 200 --seed 0 --gpu 12 --lr_schedule diminishing --lr_reduce_rate 10 --lr_iter_reduce 10000 --save_path ../../../save_async/ 2>&1 | tee ../../../save_async/cifar/0.9/4/train_logs.txt