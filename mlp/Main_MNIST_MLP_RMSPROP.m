clear all;

addpath(genpath('../CoreModules'));
n_epoch = 20;
dataset_name = 'mnist';
network_name = 'mlp';
%use_gpu = 1;
use_gpu = 0;

PrepareDataFunc = @PrepareData_MNIST_MLP;
NetInit = @net_init_mlp_mnist;

%automatically select learning rates
use_selective_sgd = 1;
%select a new learning rate every n epochs
ssgd_search_freq = 10;
learning_method = @rmsprop; %training method: @sgd, @rmsprop, @adagrad, @adam
opts.parameters.mom = 0.7;
opts.parameters.clip = 1e1;
%sgd parameter
%(unnecessary if selective-sgd is used)
sgd_lr = 5e-2;

opts.parameters.weightDecay = 0;
opts.parameters.batch_size = 500;

Main_Template();