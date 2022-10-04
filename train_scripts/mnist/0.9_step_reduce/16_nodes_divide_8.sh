#!/bin/bash
#SBATCH --time=70:00:00

#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --partition=gpulong
#SBATCH --job-name=asysonatampi16nodemnist
#SBATCH --err=myJob.err
#SBATCH --out=myJob.out

#DIR = "save_async/mnist"
#[ ! -d "$DIR" ] && mkdir -p "$DIR"

ml TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4
ml matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
ml SciPy-bundle/2019.10-fosscuda-2019b-Python-3.7.4

mpiexec -np 16 python ../../../src/main.py --iter 55000 --step 0.01 --eps 0.0001 --model cnn --dataset mnist --batch_size 128 --connectivity 0.9 --snapshot 30 --len_shards 200 --seed 0 --gpu 12 --lr_schedule step_reduce --lr_reduce_rate 8 --lr_iter_reduce 5500 --save_path ../../../save_async_step_reduce_8/ 2>&1 | tee ../../../save_async_step_reduce_8/mnist/0.9/16/train_logs.txt

