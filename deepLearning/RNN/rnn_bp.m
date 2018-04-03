function [net, res, opts] = rnn_bp(net, res, opts)

n_frames = opts.parameters.n_frames;
n_hidden_nodes = opts.parameters.n_hidden_nodes;

% calculate the gradients of the data fitting transform
for f = 1: n_frames
    opts.dzdy = res.Fit{f}(numel(net{end}.layers) + 1).dzdx;
    [net{2}, res.Fit{f}, opts] = net_bp(net{2}, res.Fit{f}, opts);
end

% bptt, calculate the gradient wrt memory cell
dzdh = 0; % accumulated gradient in later time frames
for f = n_frames: -1: 1
    % calculate the gradient in the input transform.
    dzdh = dzdh + res.Fit{f}(1).dzdx;
    opts.dzdy = dzdh;
    [net{1}, res.Input{f}, opts] = net_bp(net{1}, res.Input{f}, opts);
    % bp to previous time frame
    dzdh = opts.parameters.Id_w .* dzdh + res.Input{f}(1).dzdx(1: n_hidden_nodes, :); % an incomplete dzdh_(f-1)
end

% accumulate gradients in all time frames
res.Fit = average_gradients_in_frames(res.Fit);
res.Input = average_gradients_in_frames(res.Input);
end