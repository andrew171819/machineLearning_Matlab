clear all;

addpath(genpath('../subprograms'));
n_epoch = 20;
dataset_name = 'mnist';
network_name = 'mlp';
use_gpu = 0;

PrepareDataFunc = @prepareData_mnist_mlp;
NetInit = @net_init_mlp_mnist;

use_selective_sgd = 1;
ssgd_search_freq = 10;
learning_method = @rmsprop; % @sgd, @rmsprop, @adagrad, @adam
opts.parameters.mom = 0.7;
opts.parameters.clip = 1e1;
sgd_lr = 5e-2;

opts.parameters.weightDecay = 0;
opts.parameters.batch_size = 500;

Main_Template();