function [J, grad] = costFunction(X, y, theta, lambda)
m = length(y);
J = 0;
grad = zeros(size(theta));
htheta = X * theta;
n = size(theta);
J = 1 / (2 * m) * sum((htheta - y) .^ 2) + lambda / (2 * m) * sum(theta(2: n) .^ 2);
grad = 1 / m * X' * (htheta - y);
grad(2: n) = grad(2: n) + lambda / m * theta(2: n);
grad = grad(:);
end
