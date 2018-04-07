function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
m = length(y);
J_history = zeros(num_iters, 1);
for iter = 1: num_iters
    htheta = X * theta;
    theta_tmp = zeros(3, 1);
    for row = 1: size(theta, 1)
        theta_tmp(row) = theta(row) - alpha / m * sum((htheta - y) .* X(:, row));
    end
    theta = theta_tmp;
    J_history(iter) = computeCostMulti(X, y, theta);
end
end
