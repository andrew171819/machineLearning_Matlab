function Z = projectData(X, U, K)
Z = zeros(size(X, 1), K);
for i = 1: size(X, 1)
	x = X(i, :)';
	Z(i, :) = x' * U(:, 1: K);
end
end
