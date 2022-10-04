import tensorflow as tf
#from tensorflow.keras import datasets

import numpy as np 
import copy

def gen_CIFAR(Num_Nodes, n_train_data, n_test_data, seed, rank):
    
    cifar10 = tf.keras.datasets.cifar10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    np.random.seed(seed)
    
    p = np.random.permutation(len(x_train))
    x_train, y_train = x_train[p], y_train[p]

    p = np.random.permutation(len(x_test))
    x_test, y_test = x_test[p], y_test[p]
    
    if rank == -1:
        train_data = [x_train, y_train]
        test_data = [x_test, y_test]
    else: 
        train_data = [copy.deepcopy(x_train[rank*n_train_data:(rank+1)*n_train_data]), 
                      copy.deepcopy(y_train[rank*n_train_data:(rank+1)*n_train_data])]
    
        test_data = [copy.deepcopy(x_test[rank*n_test_data:(rank+1)*n_test_data]), 
                     copy.deepcopy(y_test[rank*n_test_data:(rank+1)*n_test_data])]
    
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(200)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(200)
    
    #del x_train, y_train
    #del x_test, y_test
    del p
    
    return [train_data, test_data, train_ds, test_ds] 

def gen_MNIST(Num_Nodes, n_train_data, n_test_data, seed, rank): 
    
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")
    
    np.random.seed(seed)
    
    p = np.random.permutation(len(x_train))
    x_train, y_train = x_train[p], y_train[p]

    p = np.random.permutation(len(x_test))
    x_test, y_test = x_test[p], y_test[p]
    
    if rank == -1:
        train_data = [x_train, y_train]
        test_data = [x_test, y_test]
    else: 
        train_data = [copy.deepcopy(x_train[rank*n_train_data:(rank+1)*n_train_data]), 
                      copy.deepcopy(y_train[rank*n_train_data:(rank+1)*n_train_data])]
    
        test_data = [copy.deepcopy(x_test[rank*n_test_data:(rank+1)*n_test_data]), 
                     copy.deepcopy(y_test[rank*n_test_data:(rank+1)*n_test_data])]
    
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(200)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(200)
    
    #del x_train, y_train
    #del x_test, y_test
    del p
    
    return [train_data, test_data, train_ds, test_ds]

def gen_FASHION_MNIST(Num_Nodes, n_train_data, n_test_data, seed, rank): 
    
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")
    
    np.random.seed(seed)
    
    p = np.random.permutation(len(x_train))
    x_train, y_train = x_train[p], y_train[p]

    p = np.random.permutation(len(x_test))
    x_test, y_test = x_test[p], y_test[p]
    
    if rank == -1:
        train_data = [x_train, y_train]
        test_data = [x_test, y_test]
    else: 
        train_data = [copy.deepcopy(x_train[rank*n_train_data:(rank+1)*n_train_data]), 
                      copy.deepcopy(y_train[rank*n_train_data:(rank+1)*n_train_data])]
    
        test_data = [copy.deepcopy(x_test[rank*n_test_data:(rank+1)*n_test_data]), 
                     copy.deepcopy(y_test[rank*n_test_data:(rank+1)*n_test_data])]
    
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(200)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(200)
    
    #del x_train, y_train
    #del x_test, y_test
    del p
    
    return [train_data, test_data, train_ds, test_ds]


#############################################################################################################
    # class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
    #                'dog', 'frog', 'horse', 'ship', 'truck']

    # plt.figure(figsize=(10,10))
    # for i in range(25):
    #     plt.subplot(5,5,i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(x_train[i], cmap=plt.cm.binary)
    #     # The CIFAR labels happen to be arrays, 
    #     # which is why you need the extra index
    #     plt.xlabel(class_names[y_train[i][0]])
    # plt.show()