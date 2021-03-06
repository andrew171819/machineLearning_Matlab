function net = net_init(opts)

rng('default');
rng(0) ;

f = 1 / 100 ;
net.layers = {} ;
net.layers{end + 1} = struct('type', 'mlp', 'weights', {{f * randn(128, 28 * 28 * 1, 'single'), zeros(128, 1, 'single')}});
net.layers{end + 1} = struct('type', 'relu');
net.layers{end + 1} = struct('type', 'mlp', 'weights', {{f * randn(128, 128, 'single'), zeros(128, 1, 'single')}});
net.layers{end + 1} = struct('type', 'relu');
net.layers{end + 1} = struct('type', 'mlp', 'weights', {{f * randn(10, 128,  'single'), zeros(10, 1, 'single')}});
net.layers{end + 1} = struct('type', 'softmaxloss');

for i = 1: numel(net.layers)
    if strcmp(net.layers{i}.type, 'mlp')
        net.layers{1,i}.momentum{1} = zeros(size(net.layers{1, i}.weights{1}));
        net.layers{1,i}.momentum{2} = zeros(size(net.layers{1, i}.weights{2}));
    end
end