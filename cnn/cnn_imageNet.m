function cnn_imageNet()
addpath(genpath('../subprograms'))
if ~exist('imagenet-vgg.mat', 'file')
    fprintf('downloading a model\n');
    urlwrite('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-f.mat', 'imagenet-vgg.mat');
end
net = load('imagenet-vgg.mat');

% obtain and preprocess an image
im = imread('test.jpg');
im_ = single(im); % 255 range
im_ = imresize(im_, net.meta.normalization.imageSize(1: 2));
im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage);

% run cnn
opts = [];
opts.use_gpu = 0;
opts.use_cudnn = 0;

opts.training = 0;
opts.use_corr = 1;
res(1).x = im_;

if opts.use_gpu
    net = SwitchProcessor(net, 'gpu');
end
tic;
[net, res, opts] = net_ff(net, res, opts);
toc;

scores = squeeze(gather(res(end).x));
[bestScore, best] = max(scores);
figure(1);
clf;
imagesc(im);
title(sprintf('%s (%d), score %.3f', net.meta.classes.description{best}, best, bestScore));
drawnow;