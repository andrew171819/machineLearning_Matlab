function [J grad] = costFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
Theta1 = reshape(nn_params(1: hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))): end), num_labels, (hidden_layer_size + 1));
m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2];
z3 = a2 * Theta2';
htheta = sigmoid(z3);
for k = 1: num_labels
    yk = y == k;
    hthetak = htheta(:, k);
    Jk = 1 / m * sum(-yk .* log(hthetak) - (1 - yk) .* log(1 - hthetak));
    J = J + Jk;
end
regularization = lambda / (2 * m) * (sum(sum(Theta1(:, 2: end) .^ 2)) + sum(sum(Theta2(:, 2: end) .^ 2)));
J = J + regularization;
for t = 1: m
    for k = 1: num_labels
        yk = y(t) == k;
        delta_3(k) = htheta(t, k) - yk;
    end
    delta_2 = Theta2' * delta_3' .* sigmoidGradient([1, z2(t, :)])';
    delta_2 = delta_2(2:end);
    Theta1_grad = Theta1_grad + delta_2 * a1(t, :);
    Theta2_grad = Theta2_grad + delta_3' * a2(t, :);
end
Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;
Theta1_grad(:, 2: end) = Theta1_grad(:, 2: end) + lambda / m * Theta1(:, 2: end);
Theta2_grad(:, 2: end) = Theta2_grad(:, 2: end) + lambda / m * Theta2(:, 2: end);
grad = [Theta1_grad(:); Theta2_grad(:)];
end