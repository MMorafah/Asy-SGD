import numpy as np 
from matplotlib.pyplot import subplots

import argparse
import copy

import tensorflow as tf

def test_args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--iter', type=int, default=1000, help="number of iterations for each Agent")
    parser.add_argument('--num_nodes', type=int, default=10, help="number of nodes")
    parser.add_argument('--batch_size', type=int, default=200, help="test batch size: B")
    parser.add_argument('--save_path', type=str, default='../save/', help="save path")
    parser.add_argument('--len_shards', type=int, default=20, help="Length of shards to load")
    parser.add_argument('--snapshot', type=int, default=30, help="Snapshot time: default 30 seconds")
    parser.add_argument('--connectivity', type=float, default=0.5, help="Connectivity of graph: default 0.5")
    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    
    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--verbose', type = int, default=100, help='verbose print at every k iterations')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    
    args = parser.parse_args()
    return args

def AVG_weight_fn(models): 
    num_weights = len(models[0])
    AVG_W = []
    for i in range(num_weights):
        w = [models[j][i] for j in range(len(models))]
        w = np.array(w).sum(axis=0) / len(models)
        #AVG_W.append(tf.constant_initializer(w))
        AVG_W.append(w)
    return AVG_W

def test_step(images, labels, model, test_loss_object, test_accuracy_object, loss_object):
    
    with tf.GradientTape() as tape:
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)
        gradients = tape.gradient(t_loss, model.trainable_variables)
    
    test_loss_object(t_loss)
    test_accuracy_object(labels, predictions)
    
    return gradients

def plot_loss(train_loss, test_loss, glob_time, savename):
    fig, ax = subplots(1,1, figsize= (6,4))
    
    ax.plot(glob_time, train_loss)
    ax.plot(glob_time, test_loss)
    
    ax.legend(['Train Loss', 'Test Loss'], loc = 'best', prop={'size':18})
    ax.set_xlabel('Time (seconds)', fontdict = {'size': 18})
    ax.set_ylabel('Loss', fontdict = {'size': 18})
    ax.set_title('Loss vs Time', fontdict = {'size': 18})
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    
    fig.savefig(savename, bbox_inches = "tight")
    return 

def plot_acc(train_acc, test_acc, glob_time, savename):
    fig, ax = subplots(1,1, figsize= (6,4))

    ax.plot(glob_time, train_acc)
    ax.plot(glob_time, test_acc)
    
    ax.legend(['Train Accuracy', 'Test Accuracy'], loc = 'lower right', prop={'size':18})
    ax.set_xlabel('Time (seconds)', fontdict = {'size': 18})
    ax.set_ylabel('Accuracy', fontdict = {'size': 18})
    ax.set_title('Accuracy vs Time', fontdict = {'size': 18})
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    
    fig.savefig(savename, bbox_inches = "tight")
    return 

def plot_consensus(err, glob_time, savename):
    fig, ax = subplots(1,1, figsize= (6,4))
        
    ax.plot(glob_time, err)

    ax.set_xlabel('Time (seconds)', fontdict = {'size': 18})
    ax.set_ylabel('Consensus Error', fontdict = {'size': 18})
    ax.set_yscale('log')
    ax.set_title('Consensus Error vs Time', fontdict = {'size': 18})
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    
    fig.savefig(savename, bbox_inches = "tight")
    return

def plot_norm_grad(norm_grads, glob_time, savename):
    fig, ax = subplots(1,1, figsize= (6,4))
        
    ax.plot(glob_time, norm_grads)

    ax.set_xlabel('Time (seconds)', fontdict = {'size': 18})
    ax.set_ylabel('Norm of Gradients', fontdict = {'size': 18})
    ax.set_yscale('log')
    ax.set_title('Norm of Gradients vs Time', fontdict = {'size': 18})
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    
    fig.savefig(savename, bbox_inches = "tight")
    return

def plot_max_delay(delay_max, glob_time, savename):
    fig, ax = subplots(1,1, figsize= (6,4))

    ax.plot(glob_time, delay_max)

    ax.set_xlabel('Time (seconds)', fontdict = {'size': 18})
    ax.set_ylabel('Max Delay (iterations)', fontdict = {'size': 18})
    ax.set_title('Max Delay vs Time', fontdict = {'size': 18})
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    
    fig.savefig(savename, bbox_inches = "tight")

    return 

def plot_time_per_iter_hist(agents_results, savename):
    num_agents = len(agents_results)
    Niter = len(agents_results[0])
    
    time_per_iter = np.zeros([int(Niter * num_agents)], dtype='float32')
    
    for i in range(len(agents_results)):
        time_per_iter[i*Niter:(i+1)*Niter] = agents_results[i][:,0]
                   
    fig, ax = subplots(1,1, figsize= (6,4))
    
    ax.hist(time_per_iter, bins = 25)

    ax.set_xlabel('Time per Iteration (seconds)', fontdict = {'size': 18})
    ax.set_ylabel('Counts', fontdict = {'size': 18})
    ax.set_title('Histogram: Time per Iteration', fontdict = {'size': 18})
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    
    fig.savefig(savename, bbox_inches = "tight")

    return 

def plot_memory_hist(agents_results, savename):
    num_agents = len(agents_results)
    Niter = len(agents_results[0])
    
    memory = np.zeros([int(Niter * num_agents)], dtype='float32')
    
    for i in range(len(agents_results)):
        memory[i*Niter:(i+1)*Niter] = agents_results[i][:,2]
                   
    fig, ax = subplots(1,1, figsize= (6,4))
    
    ax.hist(memory, bins = 25)

    ax.set_xlabel('Memory per Iteration (MB)', fontdict = {'size': 18})
    ax.set_ylabel('Counts', fontdict = {'size': 18})
    ax.set_title('Histogram: Memory per Iteration', fontdict = {'size': 18})
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    
    fig.savefig(savename, bbox_inches = "tight")

    return 

def consensus_err(agents_weights, avg_weights):
    avg = []
    for i in range(len(avg_weights)):
        avg.append(avg_weights[i].reshape([-1]))
    avg = np.hstack(avg)
    
    agents_ravel_w = []
    for i in range(len(agents_weights)):
        temp = [w.reshape([-1]) for w in agents_weights[i]]
        temp = np.hstack(temp)
        agents_ravel_w.append(copy.deepcopy(temp))
        
    diff = 0  
    s = 0
    for i in range(len(agents_weights)): 
        r = agents_ravel_w[i] - avg
        diff = np.linalg.norm(r, ord=np.inf)
        s += copy.deepcopy(diff)
        
    return s/len(agents_weights)

def save_results(results, savename):
    with open(savename, 'ab') as f:
        np.save(f, results)
    return 

def load_results(filename, cnt, length):
    results = np.zeros([length,4], dtype='float32') 
    ind = 0
    with open(filename, 'rb') as f:
        for i in range(cnt-1):
            temp = np.load(f)
            results[ind:ind+temp.shape[0]] = temp 
            ind+= temp.shape[0]
        temp = np.load(f)
        results[ind:] = temp 
    return results

def load_results_all(filename):
    with open(filename, 'rb') as f:
        results = np.load(f) 
    return results
