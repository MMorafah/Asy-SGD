import numpy as np 
from matplotlib.pyplot import subplots

import argparse
import copy
import pickle 
import psutil
import os 

from model import *

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--iter', type=int, default=1000, help="number of iterations for each Agent")
    parser.add_argument('--num_outneighbor', type=int, default=3, help="number of out-neighbors")
    parser.add_argument('--step', type=float, default=0.1, help="step size")
    parser.add_argument('--eps', type=float, default=0.0001, help="epsilon size")
    parser.add_argument('--batch_size', type=int, default=100, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--test_data', type=int, default=1000, help="number of test data for each agent")
    parser.add_argument('--train_data', type=int, default=5000, help="number of train data for each agent")
    parser.add_argument('--save_path', type=str, default='../save/', help="save path")
    parser.add_argument('--len_shards', type=int, default=20, help="Length of shards to save")
    parser.add_argument('--snapshot', type=int, default=30, help="Snapshot time: default 30 seconds")
    parser.add_argument('--connectivity', type=float, default=0.5, help="Connectivity of graph: default 0.5")
    parser.add_argument('--lr_schedule', type=str, default='diminishing', help="Type of LR Schedule: diminishing / step_reduce")
    parser.add_argument('--lr_reduce_rate', type=float, default=10.0, help="LR Reduce Rate")
    parser.add_argument('--lr_iter_reduce', type=int, default=10000, help="Iteration to Reduce LR")
    
    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', type = int, default=100, help='verbose print at every k iterations')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    
    args = parser.parse_args()
    return args

def print_init_logs(Num_Nodes, Num_outneighbor, n_train_data, n_test_data, Niter, batch_size, step, eps, dataset, model): 
    print('\n Hello to REAL ASYNCHRONOUS Decentralized Non-Convex Optimization \n')
    print('Master Node Called')
    print('Making the Setup...')
    print(f'There are {Num_Nodes} Agents')
    print(f'Num_outneighbor is {Num_outneighbor}')
    print(f'Train data points per Agent is {n_train_data}')
    print(f'Test data points per Agent is {n_test_data}')
    print(f'Iteration for each Agent is {Niter}')
    print(f'Batch size is {batch_size}')
    print(f'Step is {step}')
    print(f'Eps is {eps}')
    print(f'Training on {dataset}')
    print(f'Model is {model}')
    
    return 

def directed_graph_generator(I, N_outneighbor, seed): 
    ##########################################
    # This code generates a directed graph.
    # 
    # I:               Number of agents over the graph;
    # N_outneighbor:   Number of out-neighbors of each agent
    #########################################

    ### A circle graph ###
    ### each agent passes information into its adjacent neighbor in a cyclic 
    ### manner 
    Adj = np.diag(np.ones(I-1), -1)
    Adj[0,I-1] = 1
    # Adj = Adj + np.eye(I)

    ### Adj matrix describes the out neighbor links --> look at the columns 
    ### each column says the agent i should send information to another agent 
    ### Add edge randomly ###
    seed = copy.deepcopy(seed)
    for i in range(I):
        np.random.seed(seed)
        candidates = np.random.choice(I, N_outneighbor+1, replace=False)
        seed += 1
        counter = 0
        j       = 0
        while (counter < N_outneighbor-1): 
            if (candidates[j] != i) and (int(np.mod(candidates[j]-i,I)) != 1):
                Adj[candidates[j],i] = 1
                counter = counter + 1
            j = j + 1    

    ### C is a column stochastic matrix --> each column says the neighbor out 
    ### links from agent i 
    C = (Adj+np.eye(I))@np.diag(1/np.sum(Adj+np.eye(I), axis=0))
    ### R is a row stochastic matrix --> each row says the neighbor in links to
    ### agent i 
    R = np.diag(1/np.sum(Adj+np.eye(I), axis=1)) @ (Adj+np.eye(I))
    #print('The directed graph is generated.\n')
    # save('directed_graph.mat', 'C', 'R', 'Adj')
    return [C, R, Adj]

def unpack_graph(Num_Nodes, Adj, R, C):
    R_Nin = {i:{} for i in range(Num_Nodes)}
    C_Nout = {i:{} for i in range(Num_Nodes)}
    Nin = {i:[] for i in range(Num_Nodes)}
    Nout = {i:[] for i in range(Num_Nodes)}
    for i in range(Num_Nodes):
        for j in range(Num_Nodes):
            if Adj[i,j] != 0:
                R_Nin[i][j] = copy.deepcopy(R[i,j])
                C_Nout[i][j] = copy.deepcopy(C[i,j])
                Nin[i].append(j)
            if Adj[j,i] != 0:
                Nout[i].append(j)
        R_Nin[i][i] = copy.deepcopy(R[i,i])
        C_Nout[i][i] = copy.deepcopy(C[i,i])
        
    return [R_Nin, C_Nout, Nin, Nout]

def get_memory_usage():
    """Return the memory usage in Mo."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(10 ** 6) # In Megabyte: MB
    return mem

def test_step(images, labels, model, test_loss_object, test_accuracy_object, loss_object):

    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss_object(t_loss)
    test_accuracy_object(labels, predictions)
    
    return 

def eval_agent(model, train_ds, test_ds):
    
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    test_loss_object = tf.keras.metrics.Mean(name='test_loss_object')
    test_accuracy_object = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy_object')
        
    test_loss_object.reset_states()
    test_accuracy_object.reset_states()
            
    for images, labels in train_ds:
        test_step(images, labels, model, test_loss_object, test_accuracy_object, loss_object)
    
    train_loss = copy.deepcopy(test_loss_object.result().numpy())
    train_acc = copy.deepcopy(test_accuracy_object.result().numpy())
    
    test_loss_object.reset_states()
    test_accuracy_object.reset_states()
    
    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels, model, test_loss_object, test_accuracy_object, loss_object)
        
    test_loss = copy.deepcopy(test_loss_object.result().numpy())
    test_acc = copy.deepcopy(test_accuracy_object.result().numpy())
    
    test_loss_object.reset_states()
    test_accuracy_object.reset_states()

    return [train_loss, train_acc, test_loss, test_acc]

def plot_loss(train_loss, test_loss, Niter, verbose, savename):
    fig, ax = subplots(1,1, figsize= (6,4))
    iterations = np.arange(Niter//verbose + 1)*verbose
    ax.plot(iterations, train_loss)
    ax.plot(iterations, test_loss)
    ax.legend(['Train Loss', 'Test Loss'], loc = 'best', prop={'size':18})
    ax.set_xlabel('Iteration', fontdict = {'size': 18})
    ax.set_ylabel('Loss', fontdict = {'size': 18})
    ax.set_title('Loss vs Iteration', fontdict = {'size': 18})
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    fig.savefig(savename, bbox_inches = "tight")
    return 

def plot_acc(train_acc, test_acc, Niter, verbose, savename):
    fig, ax = subplots(1,1, figsize= (6,4))
    iterations = np.arange(Niter//verbose + 1)*verbose
    ax.plot(iterations, train_acc)
    ax.plot(iterations, test_acc)
    ax.legend(['Train Accuracy', 'Test Accuracy'], loc = 'lower right', prop={'size':18})
    ax.set_xlabel('Iteration', fontdict = {'size': 18})
    ax.set_ylabel('Accuracy', fontdict = {'size': 18})
    ax.set_title('Accuracy vs Iteration', fontdict = {'size': 18})
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    fig.savefig(savename, bbox_inches = "tight")
    return 

def save_results(results, num_shards, len_shards, savename, myname):
    for i in range(num_shards-1):
        print(f'Agent {myname} is saving results shard {i}')
        with open(savename, 'ab') as f:
            np.save(f,results[i*len_shards:(i+1)*len_shards])

    print(f'Agent {myname} is saving results shard {num_shards}')
    with open(savename, 'ab') as f:
        np.save(f,results[(num_shards-1)*len_shards:])
    return 
            