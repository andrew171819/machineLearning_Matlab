function [net, res, opts] = sgd(net, res, opts)
if ~isfield(opts.parameters, 'weightDecay')
    opts.parameters.weightDecay = 1e-4;
end

if ~isfield(opts.parameters, 'clip')
    opts.parameters.clip = 0;
end

if ~isfield(net, 'iterations') || (isfield(opts, 'reset_mom') && opts.reset_mom == 1)
    net.iterations = 0;
end

if ~isfield(opts, 'results') || ~isfield(opts.results, 'lrs')
    opts.results.lrs = [];
end
opts.results.lrs = [opts.results.lrs; gather(opts.parameters.lr)];

net.iterations = net.iterations + 1;
mom_factor = (1 - opts.parameters.mom .^ net.iterations);

for layer = 1: numel(net.layers)
    if isfield(net.layers{1, layer}, 'weights')
        if opts.parameters.clip > 0
            mask = abs(res(layer).dzdw) > opts.parameters.clip;
            res(layer).dzdw(mask) = sign(res(layer).dzdw(mask)) .* opts.parameters.clip;
            mask = abs(res(layer).dzdb) > opts.parameters.clip;
            res(layer).dzdb(mask) = sign(res(layer).dzdb(mask)) .* opts.parameters.clip;
        end
        if ~isfield(net.layers{1, layer}, 'momentum') || (isfield(opts, 'reset_mom') && opts.reset_mom == 1)
            net.layers{1, layer}.momentum{1} = zeros(size(net.layers{1, layer}.weights{1}), 'like', net.layers{1, layer}.weights{1});
            net.layers{1, layer}.momentum{2} = zeros(size(net.layers{1, layer}.weights{2}), 'like', net.layers{1, layer}.weights{2});
            opts.reset_mom = 0;
        end
        net.layers{1, layer}.momentum{1} = opts.parameters.mom .* net.layers{1, layer}.momentum{1} - (1 - opts.parameters.mom) .* res(layer).dzdw - opts.parameters.weightDecay * net.layers{1, layer}.weights{1};
        net.layers{1, layer}.weights{1} = net.layers{1, layer}.weights{1} + opts.parameters.lr * net.layers{1, layer}.momentum{1} ./ mom_factor;
        net.layers{1, layer}.momentum{2} = opts.parameters.mom .* net.layers{1, layer}.momentum{2} - (1 - opts.parameters.mom) .* res(layer).dzdb;
        net.layers{1, layer}.weights{2} = net.layers{1,layer}.weights{2} + opts.parameters.lr * net.layers{1, layer}.momentum{2} ./ mom_factor;
    end
end
end