function net = net_init()
% CNN_MNIST_LENET Initialize a CNN similar for MNIST


rng('default');
rng(0) ;

f=1/100 ;
net.layers = {} ;
%%linear
%{
%net.layers{end+1} = struct('type', 'bn') ;
net.layers{end+1} = struct('type', 'mlp', ...
                           'weights', {{f*randn(2,4, 'single'), zeros(2,1,'single')}}) ;
%}

% 2-layer net
%
%net.layers{end+1} = struct('type', 'bn') ;
net.layers{end+1} = struct('type', 'mlp', ...
                           'weights', {{f*randn(16,4, 'single'), zeros(16,1,'single')}}) ;
net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'mlp', ...
                           'weights', {{f*randn(2,16, 'single'), zeros(2,1,'single')}}) ;
%}

% 3-layer net
%net.layers{end+1} = struct('type', 'bn') ;
%{
net.layers{end+1} = struct('type', 'mlp', ...
                           'weights', {{f*randn(32,4, 'single'), zeros(32,1,'single')}}) ;
net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'mlp', ...
                           'weights', {{f*randn(32,32, 'single'), zeros(32,1,'single')}}) ;
net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'mlp', ...
                           'weights', {{f*randn(2,32, 'single'), zeros(2,1,'single')}}) ;
%}


%%leave it like this, we will evaluate the cost and derivative in another function  :p

for i=1:numel(net.layers)
    if strcmp(net.layers{i}.type,'mlp')
        net.layers{1,i}.momentum{1}=zeros(size(net.layers{1,i}.weights{1}));
        net.layers{1,i}.momentum{2}=zeros(size(net.layers{1,i}.weights{2}));
    end
end

for i=1:numel(net.layers)
    if strcmp(net.layers{i}.type,'bn')
        net.layers{1,i}.norm_stats=[];
    end
end

%}
