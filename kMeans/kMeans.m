function [centroids, idx] = kMeans(X, initial_centroids, ...
max_iters, plot_progress)
if ~exist('plot_progress', 'var') || isempty(plot_progress)
	plot_progress = false;
end
if plot_progress
	figure;
	hold on;
end
[m n] = size(X);
K = size(initial_centroids, 1);
centroids = initial_centroids;
previous_centroids = centroids;
idx = zeros(m, 1);
for i = 1: max_iters
	fprintf('k-means iteration %d/%d ...\n', i, max_iters);
	if exist('octave_version')
		fflush(stdout);
    end
    idx = findClosestCentroids(X, centroids);
    centroids = computeCentroids(X, idx, K);
end
if plot_progress
	hold off;
end
end
