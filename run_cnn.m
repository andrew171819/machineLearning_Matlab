clear all;
cd('./cnn');
disp('use a pretrained imageNet cnn model')
cnn_imageNet();
cd ..

clear all;
cd('./cnn');
disp('train a new cnn model')
cifar_cnn_sgd();
cd ..