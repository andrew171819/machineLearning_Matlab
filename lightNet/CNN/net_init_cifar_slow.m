function net = net_init_cifar(opts)
% CNN_MNIST_LENET Initialize a CNN similar for MNIST

rng('default');
rng(0) ;

f = 1 / 100 ;
net.layers = {} ;
% Block 1    
net.layers{end + 1} = struct('type', 'conv', ...
'weights', {{f  *  randn(5, 5, 3, 32, 'single'), zeros(1, 32, 'single') }}, ...
'stride', 1, ...
'pad', 2);
net.layers{end + 1} = struct('type', 'relu');
net.layers{end + 1} = struct('type', 'pool', 'pool', 3, 'stride', 2, 'pad', [0, 1, 0, 1]);

% Block 2
net.layers{end + 1} = struct('type', 'conv', ...
'weights', {{f * randn(5, 5, 32, 32, 'single'), zeros(1, 32, 'single')}}, ...
'stride', 1, ...
'pad', 2) ;
net.layers{end + 1} = struct('type', 'relu') ;
net.layers{end + 1} = struct('type', 'pool', 'pool', 3, 'stride', 2, 'pad', [0, 1, 0, 1]);

% Block 3
net.layers{end + 1} = struct('type', 'conv', ...
'weights', {{f * randn(5, 5, 32, 64, 'single'), zeros(1, 64, 'single')}}, ...
'stride', 1, ...
'pad', 2) ;
net.layers{end + 1} = struct('type', 'relu') ;
net.layers{end + 1} = struct('type', 'pool', 'pool', 3, 'stride', 2, 'pad', [0, 1, 0, 1]);

% Block 4
net.layers{end + 1} = struct('type', 'conv', ...
'weights', {{f * randn(4, 4, 64, 64, 'single'), zeros(1, 64, 'single')}}, ...
'stride', 1, ...
'pad', 0) ;
net.layers{end + 1} = struct('type', 'relu') ;

% Block 5
net.layers{end + 1} = struct('type', 'conv', ...
'weights', {{f * randn(1, 1, 64, 10, 'single'), zeros(1, 10, 'single')}}, ...
'stride', 1, ...
'pad', 0) ;
net.layers{end + 1} = struct('type', 'softmaxloss') ;
%net.layers{end + 1} = struct('type', 'pdist', 'p', 2, 'noRoot', true, 'epsilon', 1e-6);
%net.layers{end + 1} = struct('type', 'mhinge');

for i = 1: numel(net.layers)
	if strcmp(net.layers{i}.type, 'conv')
		net.layers{1, i}.momentum{1} = zeros(size(net.layers{1, i}.weights{1}));
		net.layers{1, i}.momentum{2} = zeros(size(net.layers{1, i}.weights{2}));
	end
end

if (exist('opts', 'var'))
	if strcmp(opts.parameters.learning_method, 'ew_greedy_sgd') =  = 1
		for i = 1:numel(net.layers)
			if strcmp(net.layers{i}.type, 'conv')||strcmp(net.layers{i}.type, 'mlp')
				net.layers{1, i}.lr{1} = ones(size(net.layers{1, i}.weights{1})) * opts.parameters.lr;
				net.layers{1, i}.lr{2} = ones(size(net.layers{1, i}.weights{2})) * opts.parameters.lr;
			end
		end
	end
end
