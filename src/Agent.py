from mpi4py import MPI

import tensorflow as tf 
import numpy as np 
from model import *
import time 
import copy
import gc 

tf.keras.backend.clear_session()

class Agent():
    def __init__(self, train_ds, test_ds, name, R_Nin, C_Nout, Nout, Nin, COMM, model, Num_Nodes, Max_delay, 
                 step, eps, batch_size, schedule, reduce_rate, iter_reduce):
        
        self.COMM = COMM
        
        if model == 'cnn':
            self.model = CNN()
        elif model == 'lenet':
            self.model = Lenet()
        elif model == 'resnet_18':
            self.model = resnet_18()
        else: 
            exit('Error: unrecognized model')
            
        self.train_ds = copy.deepcopy(train_ds)
        self.test_ds = copy.deepcopy(test_ds)
        self.name = name
        self.num_nodes = Num_Nodes
        self.R_Nin = copy.deepcopy(R_Nin)
        self.C_Nout = copy.deepcopy(C_Nout)
        self.Nout = copy.deepcopy(Nout)
        self.Nin = copy.deepcopy(Nin)
        self.step = copy.deepcopy(step) 
        self.eps = copy.deepcopy(eps)
        self.gamma = copy.deepcopy(step)
        self.batch_size = copy.deepcopy(batch_size)
        self.n_data = len(self.train_ds[0])
        self.n_batch = int(self.n_data//self.batch_size)
        self.batch = 0
        self.p_batch = 0 
        self.n_variables = 0
        self.trainable_variables = 0
        self.Lic = 0
        self.n_test_data = len(self.test_ds[0])
        self.test_batch = 100
        self.n_test_batch = int(self.n_test_data//self.test_batch)
        self.schedule = schedule
        self.reduce_rate = reduce_rate
        self.iter_reduce = iter_reduce
        
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        
        self.Z = 0
        self.V = []
        self.rho_buff = {i:[] for i in self.Nin}
        self.rho = []
        self.send_buff = np.zeros(10)
        self.recv_buff = {i:[] for i in self.Nin}
        self.new_layers = [] 
        self.VV = {i:[] for i in self.Nin}
        self.rrho = {i:[] for i in self.Nin}
        self.unstack = []
        
        self.recv_reqs = {}
        self.send_reqs = {}
        self.msg = {}
        
        self.shapes = []
        self.reshapes = []
        self.rrho_vectorized = [] 
        self.VV_vectorized = [] 
        self.rrho_hstack = np.zeros(10)
        self.VV_hstack = np.zeros(10)
        
        self.max_delay = Max_delay
        self.reset_ind = self.max_delay*self.num_nodes
        
        self.recv_tag = {i: 0 for i in self.Nin}
        self.send_tag = {i: 0 for i in self.Nout}
        self.send_ind = {i: [] for i in self.Nout}
        self.recv_ind = {i: [] for i in self.Nin}
        
        self.indices = np.arange(self.reset_ind, dtype='int64')
        
        self.recv_reqs = {j:[MPI.REQUEST_NULL for i in range(self.reset_ind)] for j in self.Nin}
        self.send_reqs = {j:[MPI.REQUEST_NULL for i in range(self.reset_ind)] for j in self.Nout}
        
        return 
        
    def get_config(self):
        return {'name': self.name, 'R_Nin': self.R_Nin, 'C_out': self.C_Nout, 'Nout': self.Nout, 'Nin': self.Nin,
                'Lic': self.Lic, 'V': self.V, 'rho_buff': self.rho_buff, 'rho': self.rho, 'Z':self.Z }
    
    def model_init(self):
        x, y = self.next_batch()
        self.batch = 0
        _ = self.model(x)
        
        self.n_variables = len(self.model.trainable_variables)
        self.trainable_variables = self.model.trainable_variables

        self.V = [tf.Variable(tf.zeros_like(self.trainable_variables[i])) for i in range(self.n_variables)]
        self.rho = [tf.Variable(tf.zeros_like(self.trainable_variables[i])) for i in range(self.n_variables)]
        self.Z = [tf.Variable(tf.zeros_like(self.trainable_variables[i])) for i in range(self.n_variables)]
        self.new_layers = [tf.Variable(tf.zeros_like(self.trainable_variables[i])) for i in range(self.n_variables)]
        
        self.rho_buff = {j:[tf.Variable(tf.zeros_like(self.trainable_variables[i])) for i in range(self.n_variables)] for j in self.Nin}
        
        self.rrho = {j:[tf.Variable(tf.zeros_like(self.trainable_variables[i])) for i in range(self.n_variables)] for j in self.Nin}
        
        self.VV = {j:[tf.Variable(tf.zeros_like(self.trainable_variables[i])) for i in range(self.n_variables)] for j in self.Nin}
        
        self.shapes = [w.numpy().shape for w in self.trainable_variables]
        shape = 0
        for w in self.trainable_variables:
            shape += w.numpy().reshape([-1]).shape[0]
            self.reshapes.append(shape)
            
        self.rrho_vectorized = [w.numpy().reshape([-1]) for w in self.trainable_variables] 
        self.VV_vectorized = [w.numpy().reshape([-1]) for w in self.trainable_variables] 
        self.rrho_hstack = np.hstack(self.rrho_vectorized)
        self.VV_hstack = np.hstack(self.VV_vectorized)
        
        self.msg = np.hstack([self.VV_hstack, self.rrho_hstack])
        self.send_buff = {j:copy.deepcopy(self.msg) for j in self.Nout}
        self.recv_buff = {j:copy.deepcopy(self.msg) for j in self.Nin}
        return 
        
    def next_batch(self):        
        if np.mod(self.batch, self.n_batch) == 0:
            self.batch = 0
        cc = np.arange(self.batch*self.batch_size,(self.batch+1)*self.batch_size, dtype=int)
        self.batch += 1
        return [copy.deepcopy(self.train_ds[0][cc]), copy.deepcopy(self.train_ds[1][cc])]
    
    def prev_batch(self):
        if np.mod(self.p_batch, self.n_batch) == 0:
            self.p_batch = 0
        cc = np.arange(self.p_batch*self.batch_size,(self.p_batch+1)*self.batch_size, dtype=int)
        self.p_batch += 1
        return [copy.deepcopy(self.train_ds[0][cc]), copy.deepcopy(self.train_ds[1][cc])]
    
###################################################################################   
    def init_z(self): 
        x, y = self.next_batch()
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            loss = self.loss_object(y, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)

            assert len(gradients) == self.n_variables

            for i in range(self.n_variables):
                tf.debugging.assert_all_finite(gradients[i], 'gradients has nan!', name=None)
                tf.debugging.assert_all_finite(self.Z[i], 'Z has nan!', name=None)

                tf.compat.v1.assign(self.Z[i], tf.Variable(gradients[i]))
                assert isinstance(self.Z[i], tf.Variable)
        #del tape, gradients, x, y, predictions, loss
        
        return 
    
    def local_descent(self): 
        for i in range(self.n_variables):
            ## Local Descent (S.3) -- Complete 
            temp = tf.Variable(self.model.trainable_variables[i] - self.gamma * self.Z[i]) 
            
            tf.compat.v1.assign(self.V[i], temp)
            
            tf.debugging.assert_all_finite(self.V[i], 'V has nan!', name=None) #### CHECKING
            
            ### Consensus and gradient tracking (S.4) -- In progress
            tf.compat.v1.assign(self.new_layers[i], tf.Variable(self.R_Nin[self.name]*temp))
            
            tf.debugging.assert_all_finite(self.new_layers[i], 'new_layer has nan!', name=None) #### CHECKING
            
            assert isinstance(self.V[i], tf.Variable)
            assert isinstance(self.new_layers[i], tf.Variable)
            
            del temp 
            
        return
     
    def decode_msg(self, neighbor): 
        self.VV_hstack = copy.deepcopy(self.recv_buff[neighbor][0:self.reshapes[-1]])
        self.rrho_hstack = copy.deepcopy(self.recv_buff[neighbor][self.reshapes[-1]:])
        
        ind = 0
        for i in range(self.n_variables):
            temp = copy.deepcopy(self.VV_hstack[ind:self.reshapes[i]].reshape(self.shapes[i]))
            tf.compat.v1.assign(self.VV[neighbor][i], tf.Variable(temp))
            tf.debugging.assert_all_finite(self.VV[neighbor][i], 'VV has nan!', name=None)
            
            temp = copy.deepcopy(self.rrho_hstack[ind:self.reshapes[i]].reshape(self.shapes[i]))
            tf.compat.v1.assign(self.rrho[neighbor][i], tf.Variable(temp))
            tf.debugging.assert_all_finite(self.rrho[neighbor][i], 'rrho has nan!', name=None)

            ind = self.reshapes[i]
           
        return 
    
    def init_recv_msg(self):
        for neighbor in self.Nin:
            idx = copy.deepcopy(np.delete(self.indices, self.recv_ind[neighbor])[0])
            
            tf.debugging.assert_all_finite(self.recv_buff[neighbor], 'recv_buff has nan!', name=None)
            
            self.recv_reqs[neighbor][idx] = self.COMM.Irecv([self.recv_buff[neighbor], MPI.FLOAT], source= neighbor, tag= self.recv_tag[neighbor])
            
            self.recv_tag[neighbor] += 1
            
            tf.debugging.assert_all_finite(self.recv_buff[neighbor], 'recv_buff has nan!', name=None)
            
            self.recv_ind[neighbor].append(copy.deepcopy(idx))
#             if len(self.recv_ind[neighbor]) >= 4 * (self.reset_ind//5):
#                 print(f'WARNING @ Agent {self.name}: recv_buff full 80%')
            
        return 
    
    def reset_recv_ind(self, neighbor, idx):
        data1 = np.zeros(self.msg.shape , dtype='float32') + 1e-15
        
        for i in range(idx, -1, -1):
            ii = copy.deepcopy(self.recv_ind[neighbor][i])
            if not MPI.Request.Test(self.recv_reqs[neighbor][ii]):
                MPI.Request.Cancel(self.recv_reqs[neighbor][ii])
                MPI.Request.Free(self.recv_reqs[neighbor][ii])
                
            self.recv_reqs[neighbor][ii] = MPI.REQUEST_NULL
            #self.recv_buff[neighbor] = copy.deepcopy(data1)            
            del self.recv_ind[neighbor][i]
        
        del data1
        return 
    
    def recv_msg_neighbor(self, neighbor):
        idxes = copy.deepcopy(self.recv_ind[neighbor])
        
        for i in range(len(idxes)-1, -1, -1):
            idx = copy.deepcopy(idxes[i])
            tf.debugging.assert_all_finite(self.recv_buff[neighbor], 'recv_buff has nan!', name=None)

            if MPI.Request.Test(self.recv_reqs[neighbor][idx]):
                #print(f'Agent {self.name} is receiving message from {neighbor}')
                ###############################################
                tf.debugging.assert_all_finite(self.recv_buff[neighbor], 'recv_buff has nan!', name=None)
                #if self.Lic > 1: 
                    #assert np.mean(np.isclose(self.recv_buff[neighbor][idx], data1)) != 1.0
                ##############################################
                self.decode_msg(neighbor)
                self.reset_recv_ind(neighbor, self.recv_ind[neighbor].index(idx))
                break 
            else: 
                #print(f'Agent {self.name} did not received message from {neighbor}')
                pass

        return 
    
    def sum_grads(self): 
        for neighbor in self.Nin:
            
            self.recv_msg_neighbor(neighbor)
            
            for i in range(self.n_variables):
                tf.compat.v1.assign(self.new_layers[i], tf.Variable(self.new_layers[i] + self.R_Nin[neighbor]*self.VV[neighbor][i]))
                
                assert isinstance(self.new_layers[i], tf.Variable)
                tf.debugging.assert_all_finite(self.new_layers[i], 'new_layers has nan!', name=None)
                
                ### Sum Gradients (S.5.1)
                temp = tf.Variable(self.Z[i] + self.C_Nout[neighbor] *(self.rrho[neighbor][i] - self.rho_buff[neighbor][i]))
                tf.compat.v1.assign(self.Z[i], temp) 
            
                assert isinstance(self.Z[i], tf.Variable)
                tf.debugging.assert_all_finite(self.Z[i], 'Z has nan!', name=None)
                
                assert isinstance(self.rho_buff[neighbor][i], tf.Variable)
                ### Mass-Buffer Update (S.5.3) -- Complete
                tf.compat.v1.assign(self.rho_buff[neighbor][i], tf.Variable(self.rrho[neighbor][i]))
                tf.debugging.assert_all_finite(self.rho_buff[neighbor][i], 'rho_buff has nan!', name=None)
        return 
    
    def push(self): 
        with tf.GradientTape(persistent=True) as tape:
            x,y = self.prev_batch()
            predictions1 = self.model(x, training=True)
            loss1 = self.loss_object(y, predictions1)
            gradients1 = tape.gradient(loss1, self.model.trainable_variables)
            
            #del x, y, predictions1, loss1
                
            ## Evaluation 
            #self.evaluate(x, y) 
                        
            for i in range(self.n_variables):
                tf.compat.v1.assign(self.model.trainable_variables[i], self.new_layers[i])
                assert isinstance(self.model.trainable_variables[i], tf.Variable)
                tf.debugging.assert_all_finite(self.model.trainable_variables[i], 'model variable has nan!', name=None)
                
            x,y = self.next_batch()
            predictions2 = self.model(x, training=True)
            loss2 = self.loss_object(y, predictions2)
            gradients2 = tape.gradient(loss2, self.model.trainable_variables)
            
            #del x, y, predictions2, loss2
            
            for i in range(self.n_variables):
                tf.debugging.assert_all_finite(gradients1[i], 'gradient1 has nan!', name=None)
                tf.debugging.assert_all_finite(gradients2[i], 'gradient2 has nan!', name=None)
                tf.debugging.assert_all_finite(self.Z[i], 'Z has nan!', name=None)
                
                tf.compat.v1.assign(self.Z[i], tf.Variable(self.Z[i] + gradients2[i] - gradients1[i]))
                
                assert isinstance(self.Z[i], tf.Variable)
                tf.debugging.assert_all_finite(self.Z[i], 'Z has nan!', name=None)
                ### Push (S.5.2) -- Complete 
                temp = tf.Variable(self.rho[i] + self.Z[i])
                
                tf.compat.v1.assign(self.rho[i], temp)
                tf.debugging.assert_all_finite(self.rho[i], 'rho has nan!', name=None)
                
                tf.compat.v1.assign(self.Z[i], tf.Variable(self.C_Nout[self.name] * self.Z[i]))
                assert isinstance(self.Z[i], tf.Variable)
                tf.debugging.assert_all_finite(self.Z[i], 'Z has nan!', name=None)
                            
            #del gradients1, gradients2
        #del tape
        return 
    
    def encode_msg(self):
        for i in range(self.n_variables):
            self.VV_vectorized[i] = copy.deepcopy(self.V[i].numpy().reshape([-1]))
            self.rrho_vectorized[i] = copy.deepcopy(self.rho[i].numpy().reshape([-1]))
                    
        self.rrho_hstack = copy.deepcopy(np.hstack(self.rrho_vectorized))
        self.VV_hstack = copy.deepcopy(np.hstack(self.VV_vectorized))
        
        return 
    
    def init_send_buff(self, neighbor):
        self.send_buff[neighbor] = copy.deepcopy(np.hstack([self.VV_hstack, self.rrho_hstack]))
        tf.debugging.assert_all_finite(self.send_buff[neighbor], 'send_buff has nan!', name=None)
        
        return 
    
    def send_msg(self):  
        self.encode_msg()
        for neighbor in self.Nout:
            idx = copy.deepcopy(np.delete(self.indices, self.send_ind[neighbor])[0])
            self.init_send_buff(neighbor)
            #print(f'Agent {self.name} is sending message to {neighbor}')     
            self.send_reqs[neighbor][idx] = self.COMM.Isend([self.send_buff[neighbor], MPI.FLOAT], dest= neighbor, tag= self.send_tag[neighbor])
            
            self.send_tag[neighbor] += 1
            tf.debugging.assert_all_finite(self.send_buff[neighbor], 'send_buff has nan!', name=None)
            self.send_ind[neighbor].append(copy.deepcopy(idx))
#             if len(self.send_ind[neighbor]) >= 4 * (self.reset_ind//5):
#                 print(f'WARNING @ Agent {self.name}: send_buff full 80%')
        return 
    
    def reset_buff(self):     
        data1 = np.zeros(self.msg.shape , dtype='float32') + 1e-15
        for neighbor in self.Nout: 
            idxes = copy.deepcopy(self.send_ind[neighbor])
            check = False 
            for i in range(len(idxes)-1, -1, -1):
                idx = copy.deepcopy(idxes[i])
                if check:
                    #print(f'Agent {self.name} @ SEND message to {neighbor} @ Complete')
                    self.send_reqs[neighbor][idx] = MPI.REQUEST_NULL  
                    #self.send_buff[neighbor] = copy.deepcopy(data1)
                    del self.send_ind[neighbor][i]
                elif MPI.Request.Test(self.send_reqs[neighbor][idx]): 
                    self.send_reqs[neighbor][idx] = MPI.REQUEST_NULL  
                    #self.send_buff[neighbor] = copy.deepcopy(data1)
                    del self.send_ind[neighbor][i]
                    check = True 
        del data1
        
        return 
    
    def update(self):
        
        self.reset_eval()
        
        self.Lic += 1
        
        if self.schedule == 'diminishing': 
            self.gamma = self.gamma * (1-self.eps*self.gamma)
        elif self.schedule == 'step_reduce': 
            if self.Lic % self.iter_reduce == 0:
                self.gamma = self.gamma / self.reduce_rate
        else: 
            exit('Error: unrecognized LR Schedule')
        
        if (self.Lic-1) == 0:
            self.init_z()
        
        self.local_descent()

        self.init_recv_msg()
        self.sum_grads()   

        self.push()
        self.send_msg()
        self.reset_buff()

        gc.collect()
        
        return         
#########################################################################################
    def get_len_send_buff(self):
        diction = {}
        for neighbor in self.Nout:
            diction[neighbor] = len(self.send_ind[neighbor]) / self.reset_ind
        return diction
    
    def get_len_recv_buff(self):
        diction = {}
        for neighbor in self.Nin:
            diction[neighbor] = len(self.recv_ind[neighbor]) / self.reset_ind
        return diction
    
    def get_train_loss(self):
        return self.train_loss.result().numpy()
    
    def get_test_loss(self):
        return self.test_loss.result().numpy()
    
    def get_train_acc(self):
        return self.train_accuracy.result().numpy()
    
    def get_test_acc(self):
        return self.test_accuracy.result().numpy() 
    
    def evaluate(self):
        
        #predictions = self.model(x, training=False)
        #loss = self.loss_object(y, predictions)
        #self.train_loss(loss)
        #self.train_accuracy(y, predictions)
        
        self.reset_eval()
        self.eval_testset()
        self.eval_trainset()
        template = 'Agent {}, Lic {}, Loss: {}, Train Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(self.name,
                              self.Lic,
                              self.train_loss.result(),
                              self.train_accuracy.result() * 100,
                              self.test_loss.result(),
                              self.test_accuracy.result() * 100))
        
        r = [self.train_loss.result().numpy(), self.train_accuracy.result().numpy(), self.test_loss.result().numpy(), 
             self.test_accuracy.result().numpy()]
        
        return r 
    
    def reset_eval(self): 
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()
        return 
        
    def eval_testset(self):
        for i in range(self.n_test_batch):
            x_hat = copy.deepcopy(self.test_ds[0][i*self.test_batch:(i+1)*self.test_batch])
            y_hat = copy.deepcopy(self.test_ds[1][i*self.test_batch:(i+1)*self.test_batch])
            predictions = self.model(x_hat, training=False)
            t_loss = self.loss_object(y_hat, predictions)

            self.test_loss(t_loss)
            self.test_accuracy(y_hat, predictions)
        return 
    
    def eval_trainset(self):
        for i in range(self.n_batch):
            x_hat = copy.deepcopy(self.train_ds[0][i*self.batch_size:(i+1)*self.batch_size])
            y_hat = copy.deepcopy(self.train_ds[1][i*self.batch_size:(i+1)*self.batch_size])
            predictions = self.model(x_hat, training=False)
            t_loss = self.loss_object(y_hat, predictions)

            self.train_loss(t_loss)
            self.train_accuracy(y_hat, predictions)
        return 
    
