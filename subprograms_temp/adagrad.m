function [net, res, opts] = adagrad(net, res, opts)

if ~isfield(opts.parameters, 'weightDecay')
    opts.parameters.weightDecay = 1e-4;
end

if ~isfield(opts.results, 'lrs')
    opts.results.lrs = [];
end
opts.results.lrs = [opts.results.lrs; gather(opts.parameters.lr)];

if ~isfield(opts.parameters, 'eps')
    opts.parameters.eps = 1e-6;
end

for layer = 1: numel(net.layers)
    if isfield(net.layers{1, layer},'weights')
        if ~isfield(net.layers{1, layer}, 'momentum') || (isfield(opts, 'reset_mom') && opts.reset_mom == 1)
            net.layers{1,layer}.momentum{1} = zeros(size(net.layers{1, layer}.weights{1}), 'like', net.layers{1,layer}.weights{1});
            net.layers{1,layer}.momentum{2} = zeros(size(net.layers{1, layer}.weights{2}), 'like', net.layers{1,layer}.weights{2});
            opts.reset_mom = 0;
        end
        
        net.layers{1, layer}.momentum{1} = net.layers{1, layer}.momentum{1} + res(layer).dzdw .^ 2;
        net.layers{1, layer}.weights{1} = net.layers{1, layer}.weights{1} - opts.parameters.lr * res(layer).dzdw ./ (net.layers{1, layer}.momentum{1} .^ 0.5 + opts.parameters.eps)- opts.parameters.weightDecay * net.layers{1, layer}.weights{1};
        net.layers{1, layer}.momentum{2} = net.layers{1, layer}.momentum{2} + res(layer).dzdb .^ 2;
        net.layers{1, layer}.weights{2} = net.layers{1, layer}.weights{2} - opts.parameters.lr * res(layer).dzdb ./ (net.layers{1, layer}.momentum{2} .^ 0.5 + opts.parameters.eps);
    end
end
end