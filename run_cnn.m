clear all;
cd('./cnn');
disp('test using a pretrained imageNet cnn model')
cnn_imageNet();
cd ..

clear all;
cd('./cnn');
disp('training a new cnn model')
cifar_cnn_sgd();
cd ..