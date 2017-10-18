function [net, res, opts] = adam(net, res, opts)
%NET_APPLY_GRAD_SGD Summary of this function goes here

if ~isfield(opts.parameters, 'weightDecay')
	opts.parameters.weightDecay = 0;
end

if (~isfield(opts.parameters, 'mom2'))
	opts.parameters.mom2 = 0.999; 
end

if ~isfield(net, 'iterations') || (isfield(opts, 'reset_mom') && opts.reset_mom == 1)
	net.iterations = 0;
end

if ~isfield(opts, 'results') || ~isfield(opts.results, 'lrs')
	opts.results.lrs = []; %not really necessary
end
opts.results.lrs = [opts.results.lrs; gather(opts.parameters.lr)];

if ~isfield(opts.parameters, 'eps')
	opts.parameters.eps = 1e-8;
end

net.iterations = net.iterations + 1;

for layer = 1: numel(net.layers)
	if isfield(net.layers{1, layer}, 'weights') %strcmp(net.layers{layer}.type, 'conv') || strcmp(net.layers{layer}.type, 'mlp')
		if ~isfield(net.layers{1, layer}, 'momentum') || (isfield(opts, 'reset_mom') && opts.reset_mom == 1) || length(net.layers{1, layer}.momentum)<4
			net.layers{1, layer}.momentum{1} = zeros(size(net.layers{1, layer}.weights{1}), 'like', net.layers{1, layer}.weights{1});
			net.layers{1, layer}.momentum{2} = zeros(size(net.layers{1, layer}.weights{2}), 'like', net.layers{1, layer}.weights{2});
			net.layers{1, layer}.momentum{3} = net.layers{1, layer}.momentum{1}; %initialize
			net.layers{1, layer}.momentum{4} = net.layers{1, layer}.momentum{2}; %initialize
			opts.reset_mom = 0;
		end
	end
end

mom_factor = (1 - opts.parameters.mom .^ net.iterations);
mom_factor2 = (1 - opts.parameters.mom2 .^ net.iterations);

for layer = 1: numel(net.layers)
	     if isfield(net.layers{1, layer}, 'weights')
		     net.layers{1, layer}.momentum{1} = opts.parameters.mom .* net.layers{1, layer}.momentum{1} + (1 - opts.parameters.mom) .* res(layer).dzdw;
		     net.layers{1, layer}.momentum{3} = opts.parameters.mom .* net.layers{1, layer}.momentum{3} + (1 - opts.parameters.mom) .* res(layer).dzdw .^ 2;
		     net.layers{1, layer}.weights{1} = net.layers{1, layer}.weights{1} - opts.parameters.lr * net.layers{1, layer}.momentum{1} ...
		     ./ (net.layers{1, layer}.momentum{3} .^ 0.5 + opts.parameters.eps) .* mom_factor2 ^0.5 ./ mom_factor ...
		     - opts.parameters.weightDecay * net.layers{1, layer}.weights{1};

		     net.layers{1, layer}.momentum{2} = opts.parameters.mom.*net.layers{1, layer}.momentum{2} + (1 - opts.parameters.mom).*res(layer).dzdb;
		     net.layers{1, layer}.momentum{4} = opts.parameters.mom.*net.layers{1, layer}.momentum{4} + (1 - opts.parameters.mom).*res(layer).dzdb .^ 2;
		     net.layers{1, layer}.weights{2} = net.layers{1, layer}.weights{2} - opts.parameters.lr * net.layers{1, layer}.momentum{2} ...
		     ./ (net.layers{1, layer}.momentum{4} .^0.5 + opts.parameters.eps) .* mom_factor2 ^0.5 ./ mom_factor;
	     end
end
end
