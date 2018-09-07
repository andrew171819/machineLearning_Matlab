clear all;
addpath(genpath('../subprograms'));
n_epoch = 50;
dataset_name = 'cifar';
network_name = 'cnn';
use_gpu = 0;
opts.use_cudnn = 0;
data = @prepareData_cifar_cnn;
netInit = @net_init_cifar;

use_selective_sgd = 1;
ssgd_search_freq = 20;
learning_method = @sgd; % training method, @sgd, @adagrad, @rmsprop, @adam
sgd_lr = 1e-3;

Main_Template(); % call training template