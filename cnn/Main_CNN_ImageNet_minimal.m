function Main_CNN_ImageNet_minimal()
% how to run an imageNet cnn model

% setup toolbox
addpath(genpath('../CoreModules'))
% download a pre-trained cnn from the web
if ~exist('imagenet-vgg-f.mat', 'file')
    fprintf('downloading a model\n');
    urlwrite('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-f.mat', 'imagenet-vgg-f.mat');
end
net = load('imagenet-vgg-f.mat');

% obtain and preprocess an image
im = imread('test.jpg');
im_ = single(im); % 255 range
im_ = imresize(im_, net.meta.normalization.imageSize(1: 2));
im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage);

% run cnn
opts = [];
opts.use_gpu = 0;
opts.use_cudnn = 0; % requires to compile matConvNet

opts.training = 0;
opts.use_corr = 1;
res(1).x = im_;

if opts.use_gpu
    net = SwitchProcessor(net, 'gpu');
end
tic;
[net, res, opts] = net_ff(net, res, opts);
toc;

% show the classification result
scores = squeeze(gather(res(end).x));
[bestScore, best] = max(scores);
figure(1);
clf;
imagesc(im);
title(sprintf('%s (%d), score %.3f', net.meta.classes.description{best}, best, bestScore));
drawnow;