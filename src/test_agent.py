import numpy as np 
from matplotlib.pyplot import subplots

import time
import copy
import json
import gc
import os 
import pickle
import pandas as pd 

#tf.keras.backend.set_floatx('float32')

if __name__ == "__main__":
    '''

    '''
    import tensorflow as tf 
    
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) 
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
    tf.keras.backend.clear_session()
    
    import logging
    tf.get_logger().setLevel(logging.ERROR)

    from utils_test import *
    from dataset import *
    from model import *
    
    gc.collect()
    
    args = test_args_parser()
    
    Num_Nodes = args.num_nodes
    dataset = args.dataset 
    model_name = args.model
    batch_size = args.batch_size
    save_path = args.save_path
    verbose = args.verbose
    Niter = args.iter
    len_shards = args.len_shards 
    snapshot = args.snapshot
    seed = args.seed
    connectivity = args.connectivity
    
    len_snapshots = Niter + 1
    
    if len_snapshots % len_shards == 0:
        num_shards = len_snapshots//len_shards
    else:
        num_shards = len_snapshots//len_shards + 1
        
    n_train_data = 500
    n_test_data = 100
    myname = -1
    
    start = time.time()
    print('Loading Dataset...')
    if dataset == 'mnist':
        [train_data, test_data, _, _] = gen_MNIST(Num_Nodes, n_train_data, n_test_data, seed, myname)
    elif dataset == 'fashion_mnist':
        print('Loading fashion_mnist dataset...')
        [train_data, test_data, _, _] = gen_FASHION_MNIST(Num_Nodes, n_train_data, n_test_data, seed, myname)
    elif dataset == 'cifar':
        [train_data, test_data, _, _] = gen_CIFAR(Num_Nodes, n_train_data, n_test_data, seed, myname)
    else:
        exit('Error: unrecognized dataset')
        
    result_dic = {'convergence_time': 1.1, 'consensus_error_convergence': 0.001, 'consensus_error_final': 5e-4, 
       'grad_norm_convergence': 0.002, 'grad_norm_final': 2e-4, 'max_delay': 40, 'max_node_memory': 3000, 
       'convergence_train_loss': 0.01, 'convergence_train_acc': 94, 'convergence_test_loss': 0.2, 
       'convergence_test_acc': 91, 'final_train_loss': 0.01, 'final_train_acc': 95, 'final_test_loss': 0.1, 
       'final_test_acc': 96, 'final_time': 2000}
    
    save_path = save_path + dataset + '/' + str(connectivity) + '/' + str(Num_Nodes) + '/'
    agents_results = []
    
    print('Loading agents results...')
    for node in range(Num_Nodes):
        result_all_savename = save_path + str(node) + '/'
        result_all_savename = result_all_savename + 'results' + '/' + 'AGENT_' + str(node) + '.npy' 
        agents_results.append(load_results(result_all_savename, num_shards, len_snapshots))
    
    agents_snapshots_iter = {node: [] for node in range(Num_Nodes)}
    agents_snapshots_cnt = {node: 0 for node in range(Num_Nodes)}
    agents_snapshots_global_time = {node: [] for node in range(Num_Nodes)}
    
    for node in range(Num_Nodes):
        for i in range(len_snapshots):
            if i == 0:
                agents_snapshots_cnt[node] += 1
                agents_snapshots_iter[node].append(i)
                agents_snapshots_global_time[node].append(agents_results[node][i, 1])
                
            elif agents_results[node][i, 3] == 1:
                agents_snapshots_cnt[node] += 1
                agents_snapshots_iter[node].append(i)
                agents_snapshots_global_time[node].append(agents_results[node][i, 1])
                
    num_ckp = min(list(agents_snapshots_cnt.values()))
    
    if model_name == 'cnn':
        model = CNN()
    elif model_name == 'lenet':
        model = Lenet()
    elif model_name == 'resnet_18':
        model = resnet_18()
    else:
        exit('Error: unrecognized model')
    
    _ = model(train_data[0][0:10])
    
    print('Loading the agents weights to make average models and produce the average results ...')
    
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    loss_holder = tf.keras.metrics.Mean(name='test_loss_object')
    accuracy_holder = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy_object')
        
    max_delay = 0
    converge_time = 0
    converge_iter = 0 
    result_all = np.zeros([num_ckp, 8], dtype='float32')  
    # 0:train_loss, 1:train_acc, 2:test_loss, 3:test_acc, 4:consensus_error, 5:norm_grad, 6:global_time, 7:max_delay
    
    for i in range(num_ckp):
        new_weights = []
        print(f'Loading Agents CKP {i}')
        for node in range(Num_Nodes):
            filename = save_path + str(node) + '/' + 'ckp' + '/' + 'ckp' + str(i)
            model.load_weights(filename)
            new_weights.append(copy.deepcopy(model.get_weights()))
            
        print(f'Testing AVG CKP {i}')  
        try:
            with tf.device('/device:CPU:0'):
                AVG_W = AVG_weight_fn(copy.deepcopy(new_weights))
                model.set_weights(AVG_W)
                
                grads = test_step(train_data[0], train_data[1], model, loss_holder, accuracy_holder, loss_object)
            
        except RuntimeError as e:
            print(e)
        
        result_all[i,0] = loss_holder.result().numpy()
        result_all[i,1] = accuracy_holder.result().numpy()
        
        loss_holder.reset_states()
        accuracy_holder.reset_states()
        
        temp = []
        for w in grads:
            temp.append(w.numpy().reshape([-1]))
            
        temp = np.hstack(temp)
        
        result_all[i,5] = np.linalg.norm(temp, ord=np.inf)
        
        try:
            with tf.device('/device:CPU:0'):
                _ = test_step(test_data[0], test_data[1], model, loss_holder, accuracy_holder, loss_object)
        except RuntimeError as e:
            print(e)
            
        result_all[i,2] = loss_holder.result().numpy()
        result_all[i,3] = accuracy_holder.result().numpy()
        
        loss_holder.reset_states()
        accuracy_holder.reset_states()
            
        result_all[i,4] = consensus_err(new_weights, AVG_W)
        
        result_all[i,6] = min([agents_snapshots_global_time[node][i] for node in range(Num_Nodes)])
        
        max_node = max([agents_snapshots_iter[node][i] for node in range(Num_Nodes)])
        min_node = min([agents_snapshots_iter[node][i] for node in range(Num_Nodes)])
        
        result_all[i,7] = max_node - min_node
        
        if result_all[i,7] > max_delay:
            max_delay = result_all[i,7]
            
        template = 'Iteration: {}, Global Time: {:.3f}, Train Loss: {:.5f}, Train Accuracy: {:.5f}, Test Loss: {:.5f}, Test Accuracy: {:.5f}, Consensus Err: {:.5f}, Norm Gradient: {:.5f}, Max Delay: {}'
        
        print(template.format(i, result_all[i,6], result_all[i,0], result_all[i,1]*100, result_all[i,2], result_all[i,3]*100, result_all[i,4], result_all[i,5], result_all[i,7]))
        
        if (result_all[i,4] < 1e-3) and (result_all[i,5] < 1e-3): 
            print(f'Converged @ Iteration {i}, Global Time: {result_all[i,6]}')
            converge_time = result_all[i,6]
            converge_iter = i 
            result_dic['convergence_time'] = result_all[i,6]
            result_dic['consensus_error_convergence'] = result_all[i,4]
            result_dic['grad_norm_convergence'] = result_all[i,5]
            
            result_dic['convergence_train_loss'] = result_all[i,0]
            result_dic['convergence_train_acc'] = result_all[i,1]
            result_dic['convergence_test_loss'] = result_all[i,2]
            result_dic['convergence_test_acc'] = result_all[i,3]
            
    result_dic['final_time'] = result_all[-1,6] 
    result_dic['grad_norm_final'] = result_all[-1,5]
    result_dic['consensus_error_final'] = result_all[-1,4]
    result_dic['max_delay'] = max_delay
    
    result_dic['max_node_memory'] = max([max(res[:,2]) for res in agents_results])
    result_dic['min_node_memory'] = min([min(res[:,2]) for res in agents_results])

    result_dic['max_time_per_iter'] = max([max(res[:,0]) for res in agents_results])
    result_dic['min_time_per_iter'] = min([min(res[:,0]) for res in agents_results])
    
    result_dic['final_train_loss'] = result_all[-1,0]
    result_dic['final_train_acc'] = result_all[-1,1]
    result_dic['final_test_loss'] = result_all[-1,2]
    result_dic['final_test_acc'] = result_all[-1,3]
    
    save_plot_path = save_path + 'plots' + '/' 
    if not os.path.exists(save_plot_path):
        print('Making the Plots Save dir...')
        os.makedirs(save_plot_path)
        
    result_all_savename = save_plot_path + 'Result_ALL.npy'
    result_dic_savename = save_plot_path + 'Result_dic.pickle'
    result_csv_savename = save_plot_path + 'Result_CSV.csv'
    consensus_plot_savename = save_plot_path + 'Consensus_Error.png'
    avg_loss_plot_savename = save_plot_path + 'AVG_Loss.png'
    avg_acc_plot_savename = save_plot_path + 'AVG_Acc.png'
    max_delay_plot_savename = save_plot_path + 'MAX_Delay.png'
    norm_grad_plot_savename = save_plot_path + 'Norm_Grad.png'
    memory_hist_plot_savename = save_plot_path + 'Memory_hist.png'
    time_per_iter_hist_plot_savename = save_plot_path + 'time_per_iter_hist.png'
            
    print('Plotting Results...')
    plot_consensus(result_all[:,4], result_all[:,6], consensus_plot_savename)
    plot_loss(result_all[:,0], result_all[:,2], result_all[:,6], avg_loss_plot_savename)
    plot_acc(result_all[:,1], result_all[:,3], result_all[:,6], avg_acc_plot_savename)
        
    plot_max_delay(result_all[:,7], result_all[:,6], max_delay_plot_savename)
    plot_norm_grad(result_all[:,5], result_all[:,6], norm_grad_plot_savename)
    
    plot_time_per_iter_hist(agents_results, time_per_iter_hist_plot_savename)
    plot_memory_hist(agents_results, memory_hist_plot_savename)
    #plot_time_iteration(agents_results, Niter, verbose, avg_time_iter_plot_savename)
    
    print('Saving the results...')
    save_results(result_all, result_all_savename)
    
    with open(result_dic_savename, 'wb') as fp:
        pickle.dump(result_dic, fp)
        
    final_result_csv = pd.DataFrame(result_dic, index=[Num_Nodes])
    final_result_csv.to_csv(result_csv_savename, index=False)
    
    end = time.time()
    duration = end - start 
    print(f'Converged @ Iteration {converge_iter}, Global Time: {converge_time}')
    print(f'Max Delay: {max_delay}')
    print(f'Time for Testing Process: {duration}')
