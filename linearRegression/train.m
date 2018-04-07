function [theta] = train(X, y, lambda)
initial_theta = zeros(size(X, 2), 1);
costFunction = @(t) costFunction(X, y, t, lambda);
options = optimset('maxIter', 200, 'gradObj', 'on');
theta = fmincg(costFunction, initial_theta, options);
end