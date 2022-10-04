import numpy as np 
from matplotlib.pyplot import subplots

import argparse
import copy
import pickle
import json
import pandas as pd 

def test_args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--iter', type=int, default=1000, help="number of iterations for each Agent")
    parser.add_argument('--save_path', type=str, default='../save/', help="save path")
    parser.add_argument('--len_shards', type=int, default=20, help="Length of shards to load")
    parser.add_argument('--snapshot', type=int, default=30, help="Snapshot time: default 30 seconds")
    parser.add_argument('--connectivity', type=float, default=0.5, help="Connectivity of graph: default 0.5")
    # model arguments
    #parser.add_argument('--model', type=str, default='cnn', help='model name')
    
    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    
    args = parser.parse_args()
    return args

def plot_train_loss(agents_results, Num_Nodes, savename):
    fig, ax = subplots(1,1, figsize= (6,4))
    
    for i in range(len(agents_results)):
        ax.plot(agents_results[i][:,6], agents_results[i][:,0])
        
    legends = [str(n) + ' nodes' for n in Num_Nodes]
    
    ax.legend(legends, loc = 'best', prop={'size':18})
    ax.set_xlabel('Time (seconds)', fontdict = {'size': 18})
    ax.set_ylabel('Train Loss', fontdict = {'size': 18})
    ax.set_title('Train Loss vs Time', fontdict = {'size': 18})
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    
    fig.savefig(savename, bbox_inches = "tight")
    return 

def plot_test_loss(agents_results, Num_Nodes, savename):
    fig, ax = subplots(1,1, figsize= (6,4))
    
    for i in range(len(agents_results)):
        ax.plot(agents_results[i][:,6], agents_results[i][:,2])
        
    legends = [str(n) + ' nodes' for n in Num_Nodes]
    
    ax.legend(legends, loc = 'best', prop={'size':18})
    ax.set_xlabel('Time (seconds)', fontdict = {'size': 18})
    ax.set_ylabel('Test Loss', fontdict = {'size': 18})
    ax.set_title('Test Loss vs Time', fontdict = {'size': 18})
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    
    fig.savefig(savename, bbox_inches = "tight")
    return 

def plot_train_acc(agents_results, Num_Nodes, savename):
    fig, ax = subplots(1,1, figsize= (6,4))
    
    for i in range(len(agents_results)):
        ax.plot(agents_results[i][:,6], agents_results[i][:,1])
        
    legends = [str(n) + ' nodes' for n in Num_Nodes]
    
    ax.legend(legends, loc = 'best', prop={'size':18})
    ax.set_xlabel('Time (seconds)', fontdict = {'size': 18})
    ax.set_ylabel('Train Accuracy', fontdict = {'size': 18})
    ax.set_title('Train Accuracy vs Time', fontdict = {'size': 18})
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    
    fig.savefig(savename, bbox_inches = "tight")
    return 

def plot_test_acc(agents_results, Num_Nodes, savename):
    fig, ax = subplots(1,1, figsize= (6,4))
    
    for i in range(len(agents_results)):
        ax.plot(agents_results[i][:,6], agents_results[i][:,3])
        
    legends = [str(n) + ' nodes' for n in Num_Nodes]
    
    ax.legend(legends, loc = 'best', prop={'size':18})
    ax.set_xlabel('Time (seconds)', fontdict = {'size': 18})
    ax.set_ylabel('Test Accuracy', fontdict = {'size': 18})
    ax.set_title('Test Accuracy vs Time', fontdict = {'size': 18})
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    
    fig.savefig(savename, bbox_inches = "tight")
    return 

def plot_consensus(agents_results, Num_Nodes, savename):
    fig, ax = subplots(1,1, figsize= (6,4))
        
    for i in range(len(agents_results)):
        ax.plot(agents_results[i][:,6], agents_results[i][:,4])
    
    legends = [str(n) + ' nodes' for n in Num_Nodes]
    
    ax.legend(legends, loc = 'best', prop={'size':18})
    ax.set_xlabel('Time (seconds)', fontdict = {'size': 18})
    ax.set_ylabel('Consensus Error', fontdict = {'size': 18})
    ax.set_yscale('log')
    ax.set_title('Consensus Error vs Time', fontdict = {'size': 18})
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    
    fig.savefig(savename, bbox_inches = "tight")
    return

def plot_norm_grad(agents_results, Num_Nodes, savename):
    fig, ax = subplots(1,1, figsize= (6,4))
        
    for i in range(len(agents_results)):
        ax.plot(agents_results[i][:,6], agents_results[i][:,5])
    
    legends = [str(n) + ' nodes' for n in Num_Nodes]
    
    ax.legend(legends, loc = 'best', prop={'size':18})

    ax.set_xlabel('Time (seconds)', fontdict = {'size': 18})
    ax.set_ylabel('Norm of Gradients', fontdict = {'size': 18})
    ax.set_yscale('log')
    ax.set_title('Norm of Gradients vs Time', fontdict = {'size': 18})
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    
    fig.savefig(savename, bbox_inches = "tight")
    return

def plot_max_delay(agents_dics, Num_Nodes, savename):
    fig, ax = subplots(1,1, figsize= (6,4))
    delay = []
    for i in range(len(agents_dics)):
        delay.append(agents_dics[i]['max_delay'])
        
    ax.plot(Num_Nodes, delay)

    ax.set_xlabel('Number of Nodes', fontdict = {'size': 18})
    ax.set_ylabel('Max Delay (iterations)', fontdict = {'size': 18})
    ax.set_title('Max Delay vs Number of Nodes', fontdict = {'size': 18})
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    
    fig.savefig(savename, bbox_inches = "tight")

    return 

def plot_time_per_iter(agents_dics, Num_Nodes, savename):
    time_per_iter = []
    for i in range(len(agents_dics)):
        time_per_iter.append(agents_dics[i]['max_time_per_iter'])
                   
    fig, ax = subplots(1,1, figsize= (6,4))
    
    ax.plot(Num_Nodes, time_per_iter)

    ax.set_xlabel('Number of Nodes', fontdict = {'size': 18})
    ax.set_ylabel('Max Time per Iteration (seconds)', fontdict = {'size': 18})
    ax.set_title('Number of Nodes vs Max Time per Iteration', fontdict = {'size': 18})
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    
    fig.savefig(savename, bbox_inches = "tight")

    return 

def plot_memory(agents_dics, Num_Nodes, savename):
    memory = []
    for i in range(len(agents_dics)):
        memory.append(agents_dics[i]['max_node_memory'])
                   
    fig, ax = subplots(1,1, figsize= (6,4))
    
    ax.plot(Num_Nodes, memory)

    ax.set_xlabel('Number of Nodes', fontdict = {'size': 18})
    ax.set_ylabel('Max Memory per Iteration (MB)', fontdict = {'size': 18})
    ax.set_title('Number of Nodes vs Max Memory per Iteration', fontdict = {'size': 18})
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    
    fig.savefig(savename, bbox_inches = "tight")

    return 

def plot_speedup(agents_dics, Num_Nodes, savename):
    time_convergence = []
    for i in range(len(agents_dics)):
        time_convergence.append(agents_dics[i]['convergence_time'])
    
    t0 = copy.deepcopy(time_convergence[0])
    speedup = [t0/t for t in time_convergence]
    
    fig, ax = subplots(1,1, figsize= (6,4))
    
    ax.plot(Num_Nodes, speedup)

    ax.set_xlabel('Number of Nodes', fontdict = {'size': 18})
    ax.set_ylabel('Speedup', fontdict = {'size': 18})
    ax.set_title('Speedup vs Number of Nodes', fontdict = {'size': 18})
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    
    fig.savefig(savename, bbox_inches = "tight")
    
    return 

def plot_norm_grad_convergence(agents_dics, Num_Nodes, savename):
    norm_grad = []
    for i in range(len(agents_dics)):
        norm_grad.append(agents_dics[i]['grad_norm_convergence'])
                   
    fig, ax = subplots(1,1, figsize= (6,4))
    
    ax.plot(Num_Nodes, norm_grad)

    ax.set_xlabel('Number of Nodes', fontdict = {'size': 18})
    ax.set_ylabel('Norm of Gradients', fontdict = {'size': 18})
    ax.set_title('Norm of Gradients vs Number of Nodes', fontdict = {'size': 18})
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    
    fig.savefig(savename, bbox_inches = "tight")
    return 

def plot_consensus_convergence(agents_dics, Num_Nodes, savename):
    consensus = []
    for i in range(len(agents_dics)):
        consensus.append(agents_dics[i]['consensus_error_convergence'])
                   
    fig, ax = subplots(1,1, figsize= (6,4))
    
    ax.plot(Num_Nodes, consensus)

    ax.set_xlabel('Number of Nodes', fontdict = {'size': 18})
    ax.set_ylabel('Consensus Error', fontdict = {'size': 18})
    ax.set_title('Consensus Error vs Number of Nodes', fontdict = {'size': 18})
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    
    fig.savefig(savename, bbox_inches = "tight")
    return 

def load_results_all(filename):
    with open(filename, 'rb') as f:
        results = np.load(f)
    return results

def load_dic(filename):
    with open(filename, 'rb') as fp:
        data = pickle.load(fp)
    return data 
