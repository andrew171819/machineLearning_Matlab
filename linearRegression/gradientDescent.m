function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
m = length(y);
J_history = zeros(num_iters, 1);
for iter = 1: num_iters
    htheta = X * theta;
    theta0 = theta(1) - alpha / m * sum((htheta - y) .* X(:, 1));
    theta1 = theta(2) - alpha / m * sum((htheta - y) .* X(:, 2));
    theta = [theta0; theta1];
    J_history(iter) = computeCost(X, y, theta);
end
end