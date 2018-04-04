clear all;
addpath(genpath('../CoreModules'));
n_epoch = 50;
dataset_name = 'cifar';
network_name = 'slow-cnn';
use_gpu = 1;
opts.use_cudnn = 0; % requires to compile matConvNet
PrepareDataFunc = @PrepareData_CIFAR_CNN;
NetInit = @net_init_cifar_slow;

% automatically select learning rates
use_selective_sgd = 1;
% select a new learning rate every n epochs
ssgd_search_freq = 20;

learning_method = @sgd; % training method, @sgd, @adagrad, @rmsprop, @adam

sgd_lr = 1e-3;

Main_Template(); % call training template