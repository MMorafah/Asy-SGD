import numpy as np 
from matplotlib.pyplot import subplots

import time
import copy
import pickle
import pandas as pd
import gc 
import os 

if __name__ == "__main__":
    '''
    '''
    
    from utils_test_all_nodes import *    
    
    args = test_args_parser()
    
    dataset = args.dataset 
    #model_name = args.model
    save_path = args.save_path
    Niter = args.iter
    seed = args.seed
    connectivity = args.connectivity
    
    Num_Nodes = [2, 4, 8, 16, 24, 32] 
    
    start = time.time()
    
    gc.collect()
    
    save_path = save_path + dataset + '/' + str(connectivity) + '/' 
    agents_results = []
    agents_dics = []
    print('Loading agents results...')
    for node in Num_Nodes:
        result_all_savename = save_path + str(node) + '/' + 'plots' + '/'  + 'Result_ALL.npy' 
        result_dic_savename = save_path + str(node) + '/' + 'plots' + '/'  + 'Result_dic.pickle'
        agents_results.append(load_results_all(result_all_savename))
        agents_dics.append(load_dic(result_dic_savename))
          
    print('Making the dictionary to produce the plots ...')
    final_result_dic = {}
    for key in agents_dics[0].keys():
        final_result_dic[key] = []
        for dic in agents_dics:
            final_result_dic[key].append(dic[key])
        
    save_plot_path = save_path + 'plots' + '/' 
    if not os.path.exists(save_plot_path):
        print('Making the Plots Save dir...')
        os.makedirs(save_plot_path)
        
    consensus_plot_savename = save_plot_path + 'Consensus_Error.png'
    train_loss_plot_savename = save_plot_path + 'Train_Loss.png'
    test_loss_plot_savename = save_plot_path + 'Test_Loss.png'
    train_acc_plot_savename = save_plot_path + 'Train_Acc.png'
    test_acc_plot_savename = save_plot_path + 'Test_Acc.png'
    max_delay_plot_savename = save_plot_path + 'MAX_Delay.png'
    norm_grad_plot_savename = save_plot_path + 'Norm_Grad.png'
    memory_plot_savename = save_plot_path + 'Memory.png'
    time_per_iter_plot_savename = save_plot_path + 'time_per_iter.png'
    speedup_plot_savename = save_plot_path + 'Speedup.png'
            
    print('Plotting Results...')
    plot_consensus(agents_results, Num_Nodes, consensus_plot_savename)
    plot_norm_grad(agents_results, Num_Nodes, norm_grad_plot_savename)
    
    plot_train_loss(agents_results, Num_Nodes, train_loss_plot_savename)
    plot_train_acc(agents_results, Num_Nodes, train_acc_plot_savename)
    plot_test_loss(agents_results, Num_Nodes, test_loss_plot_savename)
    plot_test_acc(agents_results, Num_Nodes, test_acc_plot_savename)
    
    plot_speedup(agents_dics, Num_Nodes, speedup_plot_savename)
    
    plot_max_delay(agents_dics, Num_Nodes, max_delay_plot_savename)
    
    plot_time_per_iter(agents_dics, Num_Nodes, time_per_iter_plot_savename)
    plot_memory(agents_dics, Num_Nodes, memory_plot_savename)
    
    plot_consensus_convergence(agents_dics, Num_Nodes, consensus_plot_savename)
    plot_norm_grad_convergence(agents_dics, Num_Nodes, norm_grad_plot_savename)
    
    print('Saving the results...')
    result_dic_savename = save_plot_path + 'Result_dic.pickle'
    with open(result_dic_savename, 'wb') as fp:
        pickle.dump(final_result_dic, fp)
    
    result_csv_savename = save_plot_path + 'Result_CSV.csv'
    final_result_csv = pd.DataFrame(final_result_dic, index=Num_Nodes)
    final_result_csv.to_csv(result_csv_savename, index=False)
    
    end = time.time()
    duration = end - start 
    print(f'Time for Testing Process: {duration}')
    
