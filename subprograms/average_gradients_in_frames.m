function [accum_grad] = average_gradients_in_frames(frames)

n_frames = length(frames);
accum_grad = [];
for layer = 1: length(frames{1})
    if isfield(frames{1}(layer), 'dzdw') && ~isempty(frames{1}(layer).dzdw)
        accum_grad(layer).dzdw = 0;
        for f = 1: n_frames
            accum_grad(layer).dzdw = accum_grad(layer).dzdw + frames{f}(layer).dzdw;
        end
        accum_grad(layer).dzdw = accum_grad(layer).dzdw ./ n_frames;
    end
    
    if isfield(frames{1}(layer), 'dzdb') && ~isempty(frames{1}(layer).dzdb)
        accum_grad(layer).dzdb = 0;
        for f = 1: n_frames
            accum_grad(layer).dzdb = accum_grad(layer).dzdb + frames{f}(layer).dzdb;
        end
        accum_grad(layer).dzdb = accum_grad(layer).dzdb ./ n_frames;
    end
end
end