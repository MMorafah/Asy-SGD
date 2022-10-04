# Decentralized Asynchronous Non-convex Stochastic Optimization on Directed Graphs
The official code for the paper ''[Decentralized Asynchronous Non Convex Stochastic Optimization on Directed Graphs](https://arxiv.org/abs/2110.10406)'' **[Accepted to IEEE Transactions on Control of Network Systems, Oct 2022]**

## Usage
First, you need to run the train script. When the train script has been successfully finished, you need to run the test scripts. 

* **Training script**
Here is one example to run train scripts code for cifar10, 2nodes using cpu:
```
#SBATCH --time=70:00:00

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --partition=cpulong
#SBATCH --job-name=asysonatampi2nodecifar
#SBATCH --err=TRAIN_cifar10_2nodes.err
#SBATCH --out=TRAIN_cifar10_2nodes.out

ml TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4
ml matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
ml SciPy-bundle/2019.10-fosscuda-2019b-Python-3.7.4

dir='../../../save_async_step_reduce/cifar/0.7/2'
if [ ! -e $dir ]; then
mkdir -p $dir
fi 

mpiexec -np 2 python ../../../src/main.py --iter 35000 \
--step 0.2 \
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
--lr_reduce_rate 10 \
--lr_iter_reduce 10000 \
--save_path ../../../save_async_step_reduce/ \
2>&1 | tee $dir'/train_logs.txt'
```

| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `model` | The model architecture. Options:`resnet_18`, `cnn`, `lenet`. Default = `cnn`. |
| `dataset`      | Dataset to use. Options: `mnist`, `cifar`, `fashion_mnist`. Default = `mnist`. |
| `step` | The initial Learning rate or step size. Default = `0.1`. |
| `lr_schedule` | Learning rate reduction schedule. Options: `diminishing`, `step_reduce`. default = `diminishing`. |
| `batch-size` | Batch size, default = `100`. |
| `iter` | Number of local iterations each node perform, default = `1000`. |
| `eps` | Epsilon size for diminishing learning rate schedule, default = `0.0001`. |
| `connectivity` | Graph Connectivity percentage, default = `0.5`. |
| `snapshot`    | Snapshots intervals in seconds, default = `30`. |
| `len_shards`    | Length of saving shards. Default = `20` |
| `lr_reduce_rate` | The rate of learning rate reduction for step_reduce learning rate schedule, default = `10.0`. |
| `lr_iter_reduce` | The learning rate reduction intervals for step_reduce learning rate schedule, default = `10000`. |
| `gpu` | The id of gpu device if -1 it uses cpu, default = `0`. |
| `save_path` | Path to save snapshots, default = `../save/`. |
| `num_outneighbor` | Number of out-neighbors, default = `3`. |
| `seed` | The random seed for creating graph and partitioning dataset amongst nodes, default = `0`. |

<br> </br>
* **Test script**
Here is one example to run test scripts code for cifar10, 2nodes using cpu:
```
#!/bin/bash
#SBATCH --time=70:00:00

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH --partition=cpulong
#SBATCH --job-name=asysonatampi2nodemnist
#SBATCH --err=TEST_cifar10_2nodes.err
#SBATCH --out=TEST_cifar10_2nodes.out

ml TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4
ml matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
ml SciPy-bundle/2019.10-fosscuda-2019b-Python-3.7.4

dir='../../../save_async_step_reduce/cifar/0.7/2'

python ../../../src/test_agent.py --num_nodes 2 \
--iter 35000 \
--model resnet_18 \
--dataset cifar \
--batch_size 200 \
--connectivity 0.7 \
--snapshot 30 \
--seed 0 \
--len_shards 200 \
--save_path ../../../save_async_step_reduce/ \
2>&1 | tee $dir'/test_logs.txt'
```

| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `model` | The model architecture. Options:`resnet_18`, `cnn`, `lenet`. Default = `cnn`. |
| `dataset`      | Dataset to use. Options: `mnist`, `cifar`, `fashion_mnist`. Default = `mnist`. |
| `batch-size` | Batch size for test, default = `100`. |
| `iter` | Number of local iterations each node perform, default = `1000`. |
| `connectivity` | Graph Connectivity percentage, default = `0.5`. |
| `snapshot`    | Snapshots intervals in seconds, default = `30`. |
| `len_shards`    | Length of saving shards. Default = `20` |
| `gpu` | The id of gpu device if -1 it uses cpu, default = `0`. |
| `save_path` | Path to save snapshots, default = `../save/`. |
| `num_nodes` | Number of Nodes, default = `10`. |
| `seed` | The random seed for creating graph and partitioning dataset amongst nodes, default = `0`. |

## How to use the codes 
The codes have beeing organized as follow: 
* **src**: the src code which contains all the codes. 
* **train_scripts**: all the training scripts. for each dataset, and connectivity and different number of nodes.
* **test_scripts**: all the test scripts. for each dataset, and connectivity and different number of nodes.

Please follow these step to use the code: 
* **Training**: First, please run the train script for the corresponding dataset, connectivity and number of nodes. Please also make sure that the saving path of the results are correct. It means you need to check the `dir='../../../save_async_step_reduce/cifar/0.7/2'`, `--save_path ../../../save_async_step_reduce/` in the train scripts. Also, you can modify the `#Sbatch ....` to include gpu instead of running on cpu. To use gpu please put `#SBATCH --partition=gpulong` and `#SBATCH --gres=gpu:1`. To use CPU you only need to put `#SBATCH --partition=cpulong`. 
* **Testing**: After the training has been successfully finished, please run the test script for the corresponding dataset, connectivity, and number of nodes. Please also make sure that the saving path of the results are correct. It means you need to check the `dir='../../../save_async_step_reduce/cifar/0.7/2'`, `--save_path ../../../save_async_step_reduce/` in the test scripts and it is the same as what has been put in its corresponding train script. 

## Additional Notes 
* If you run the test script and you got an error of ... did not found, it means that the train script has not been successfully finished. When the train script or test script has been successfully finished, `Job Concluded` should exist in the logs files. 
* After running the test script, there will be a folder named `plot` in the save_path directory which contains the results. Other folders in the directory, they contain the snapshots for different nodes and they are raw saved snapshots. 
* Please also note that for cifar10, the snapshots size are huge and it may require more that 4TB of space. If there is not enough space, after running for several days the runs will be stucked! Also, the test scripts needed to be run over cpuextralong just to be safe. Sometimes the test scripts take about 1 week, especially for 8 nodes and higher. 
* To reduce the size of saved snapshots, you can increase the snapshot intervals when running the training scripts by adjusting `snapshot` argument to 90 or more seconds.

## Citation
Please cite our work if you find this repository and the paper useful. 

```
@article{kungurtsev2021decentralized,
  title={Decentralized Asynchronous Non-convex Stochastic Optimization on Directed Graphs},
  author={Kungurtsev, Vyacheslav and Morafah, Mahdi and Javidi, Tara and Scutari, Gesualdo},
  journal={arXiv preprint arXiv:2110.10406},
  year={2021}
}
```
