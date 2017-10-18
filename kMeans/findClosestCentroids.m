function idx = findClosestCentroids(X, centroids)
K = size(centroids, 1);
idx = zeros(size(X, 1), 1);
for i = 1: length(X)
	distance = inf;
	for j = 1: K
		kDistance = norm(X(i, :) - centroids(j, :));
		if (kDistance < distance)
			distance = kDistance;
			idx(i) = j;
	end
end
end
end
