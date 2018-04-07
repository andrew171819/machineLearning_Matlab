function [J, grad] = costFunctionReg(theta, X, y, lambda)
m = length(y);
J = 0;
grad = zeros(size(theta));
htheta = sigmoid(X * theta);
J = 1 / m * sum(-y .* log(htheta) - (1 - y) .* log(1 - htheta)) + lambda / (2 * m) * sum(theta(2: end) .^ 2);
grad(1) = 1 / m * sum((htheta - y) .* X(:, 1));
for i = 2: size(theta, 1)
    grad(i) = 1 / m * sum((htheta - y) .* X(:, i)) + lambda / m * theta(i);
end
end