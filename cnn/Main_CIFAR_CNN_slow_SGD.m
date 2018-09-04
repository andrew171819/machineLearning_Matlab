clear all;
addpath(genpath('../subprograms'));
n_epoch = 50;
dataset_name = 'cifar';
network_name = 'slow-cnn';
use_gpu = 0;
opts.use_cudnn = 0;
PrepareDataFunc = @PrepareData_CIFAR_CNN;
NetInit = @net_init_cifar_slow;

% automatically select learning rates
use_selective_sgd = 1;
% select a new learning rate every n epochs
ssgd_search_freq = 20;

learning_method = @sgd; % training method, @sgd, @adagrad, @rmsprop, @adam

sgd_lr = 1e-3;

Main_Template(); % call training template