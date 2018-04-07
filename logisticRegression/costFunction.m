function [J, grad] = costFunction(theta, X, y)
m = length(y);
J = 0;
grad = zeros(size(theta));
htheta = sigmoid(X * theta);
J = 1 / m * sum(-y .* log(htheta) - (1 - y) .* log(1 - htheta));
for i = 1: size(theta, 1)
    grad(i) = 1 / m * sum((htheta - y) .* X(:, i));
end
end