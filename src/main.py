import mpi4py 
from mpi4py import MPI 

import tensorflow as tf
import numpy as np
import time
import os
import psutil
import gc 
import copy

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) 
tf.keras.backend.clear_session()

#device = cuda.get_current_device()
#device.reset()

###################################### GPU ALLOCATION: Using Multi GPU ###################################
tf.debugging.set_log_device_placement(True)

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
    
if __name__ == "__main__":
    
    from utils_train import * 
    from dataset import *
    from Agent import *
    
    args = args_parser()
    
    Num_Nodes       = SIZE                     # network size 
    Num_outneighbor = int(np.floor(args.connectivity * Num_Nodes))     # num out neighbors 
    
    if Num_outneighbor == 0:
        Num_outneighbor += 1 
        
    #n_train_data    = args.train_data         # number of training data per agent 
    #n_test_data     = args.test_data          # number of test data per agent 
    
    Niter = args.iter                          # number of iterations per agent 
    
    step = args.step                           # step size 
    eps = args.eps                             # eps size 
    
    batch_size = args.batch_size               # batch size per agent 
    verbose = args.verbose                     # verbose iterations 
    
    seed = args.seed                           # random seed 
    
    dataset = args.dataset                     # name of dataset: default mnist 
    
    model_name = args.model                    # name of model: defaul cnn 
    
    save_path = args.save_path                 # Save dir: default ./save/
    
    len_shards = args.len_shards               # Length of shards to save 
    
    time_snapshot = args.snapshot              # snapshot time: default 30 seconds 
    
    connectivity = args.connectivity           # Graphs Connectivity: default 0.5 
    
    gpu = args.gpu                             # Number of GPUs 
    
    schedule = args.lr_schedule                # Type of LR Schedule
    
    reduce_rate = args.lr_reduce_rate          # LR Reduce Rate 
    
    iter_reduce = args.lr_iter_reduce             # Iterations to Reduce LR 
        
    if dataset == 'mnist' or dataset == 'fashion_mnist':
        n_train_data    = 60000 // Num_Nodes          # number of training data per agent 
        n_test_data     = 10000 // Num_Nodes          # number of test data per agent 
    elif dataset == 'cifar':
        n_train_data    = 50000 // Num_Nodes          # number of training data per agent 
        n_test_data     = 10000 // Num_Nodes          # number of test data per agent 
    ##########################################################################################
    ## Setting the setups For each Agent
    
    save_path = save_path + dataset + '/' + str(connectivity) + '/' + str(Num_Nodes) + '/' + str(RANK) + '/'
    save_path_results = save_path + 'results' + '/'
    save_path_ckp = save_path + 'ckp' + '/'
    
    if RANK == 0: 
        print(f'---------GPU AVAILABILITY---------- \n GPU AVAILABLE: {tf.test.is_gpu_available()}, CUDA: {tf.test.is_built_with_cuda()}, GPU NAME: {tf.test.gpu_device_name()}')
        
        print_init_logs(Num_Nodes, Num_outneighbor, n_train_data, n_test_data, Niter, batch_size, step, eps, dataset, model_name)
        gc.collect()
        
    ## Making the save dir   
    if not os.path.exists(save_path):
        print('Making the Save dir...')
        os.makedirs(save_path)
        os.makedirs(save_path_results)
        os.makedirs(save_path_ckp)
        
    myname = RANK
    
    result_all_savename = save_path_results + 'AGENT_' + str(myname) + '.npy'
    loss_savename = save_path_results + 'AGENT_' + str(myname) + '_loss_plot.png'
    acc_savename = save_path_results + 'AGENT_' + str(myname) + '_acc_plot.png'
     
    print(f'Agent {myname} has been called')
    
    ## Getting the Dataset for the Agents     
    if dataset == 'mnist':
        [train_data, test_data, train_ds, test_ds] = gen_MNIST(Num_Nodes, n_train_data, n_test_data, seed, myname)
    elif dataset == 'fashion_mnist':
        [train_data, test_data, train_ds, test_ds] = gen_FASHION_MNIST(Num_Nodes, n_train_data, n_test_data, seed, myname)
    elif dataset == 'cifar':
        [train_data, test_data, train_ds, test_ds] = gen_CIFAR(Num_Nodes, n_train_data, n_test_data, seed, myname)
    else:
        exit('Error: unrecognized dataset')

    print(f'Agent {myname} creating the GRAPH')
    
    [C, R, Adj] = directed_graph_generator(Num_Nodes, Num_outneighbor, seed)

    [R_Nin, C_Nout, Nin, Nout] = unpack_graph(Num_Nodes, Adj, R, C)
    
    if myname == 0:
        print(f'R_Nin = {R_Nin} \n C_Nout = {C_Nout} \n Nin = {Nin} \n Nout = {Nout}')

    Max_delay = 50
    
    print(f'Agent {myname} creating the OPTIMIZATION Node')
    
    mygpu = int(np.floor(myname/gpu))
    device = '/device:GPU:' + str(mygpu)
    
    try:
        # Specify an invalid GPU device
        with tf.device(device):
            OPT_NODE = Agent(train_data, test_data, myname, copy.deepcopy(R_Nin[myname]), copy.deepcopy(C_Nout[myname]), copy.deepcopy(Nout[myname]), copy.deepcopy(Nin[myname]), COMM, model_name, Num_Nodes, Max_delay, step, eps, batch_size, schedule, reduce_rate, iter_reduce) 
            
            print(f'Agent {myname} INIT the OPTIMIZATION Node')
            OPT_NODE.model_init()
            
    except RuntimeError as e:
        print(e)
    
    print(f'Agent {myname} Is Waiting to Start')
    
    result_all = np.zeros([Niter + 1, 4], dtype='float32') 
    # Result: Time per Iteration, Global Time, Memory, Snapshot 
    
    COMM.Barrier()
    
    tic = time.time()
    
    global_time = 0
    
    weight_name = 'ckp'
    
    save_weight_pth = save_path_ckp + weight_name + str(0)
    
    OPT_NODE.model.save_weights(save_weight_pth, save_format='h5')
    
    toc = time.time() 
        
    global_time = global_time + toc - tic 
    
    result_all[0,0] = toc - tic 
    result_all[0,1] = global_time 
    result_all[0,2] = get_memory_usage()
    
    COMM.Barrier()
    print(f'Agent {myname} Initiated and started training...')
    
    d = 0
    snap = 0
    tictic = MPI.Wtime()
    START = MPI.Wtime()
    for i in range(Niter):  
        tic_per_iter = MPI.Wtime()
        
        try:
            with tf.device(device):
                OPT_NODE.update()              
        except RuntimeError as e:
            print(e)

        gc.collect()
        
        mem = get_memory_usage()
        
        toc_per_iter = MPI.Wtime()
        time_per_iter = toc_per_iter - tic_per_iter 
        
        global_time = global_time + time_per_iter
        
        d += time_per_iter 
        
        result_all[i+1,0] = time_per_iter
        result_all[i+1,1] = global_time 
        result_all[i+1,2] = mem
        
        #toc = MPI.Wtime()
        #d = toc - tictic
        #print(f'Agent {myname} @ Iteration {i+1}, Time per Iter: {time_per_iter}, Global Time: {global_time}, d: {d}')
        if abs(d - time_snapshot) <= time_per_iter:
            snap += 1 
            
            if myname == 0:
                print(f'---------------------@@ Snapshot {snap} @@------------------------')
                
            #print(f'---EVAL--- Iteration @ {i+1}, Agent {myname}, Time: {tt}, memory usage = {get_memory_usage():.3f} Mo') 
            template = '@ Snapshot {} -- Agent {} @ Iteration {} is Taking Snapshot... Time: {:.3f} s, Global Time: {:.3f}, memory usage: {:.3f} MB'
            print(template.format(snap, myname, i+1, d, global_time, mem)) 
            
            result_all[i+1,3] = 1           

            save_weight_pth = save_path_ckp + weight_name + str(snap)
                
            OPT_NODE.model.save_weights(save_weight_pth, save_format='h5')
            
#             toc = time.time()
#             dd = toc - tictic 
#             global_time = global_time + dd
            
#             result_all[i+1,1] = global_time

            #tictic = MPI.Wtime()
            d = 0 
    
    toc = time.time()
    t = toc - tic

    print(f'Agent {myname} finished job!, Total Time {t}')
    
    COMM.Barrier()
    
    total_len = Niter + 1 
    num_shards = total_len//len_shards
    
    if total_len % len_shards != 0:
        num_shards += 1 
        
    print(f'Agent {myname} is saving the results shards now...')
    save_results(result_all, num_shards, len_shards, result_all_savename, myname)
    
    COMM.Barrier()
            
    END = MPI.Wtime()  
    
    T = END - START 
    
    print(f'JOB CONCLUDED! time: {T}')
