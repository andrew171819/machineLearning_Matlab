clear all;
cd('./MLP');
disp('testing training a multilayer perceptron')
Main_MNIST_MLP_RMSPROP();
cd ..

clear all;
cd('./ReinforcementLearning');
disp('testing training a q-network')
Main_Cart_Pole_Q_Network
cd ..

clear all;
cd('./RNN');
disp('testing training an lstm')
Main_Char_RNN();
cd ..

clear all;
cd('./CNN');
disp('testing using a pretrained imageNet convolutional neural network model')
Main_CNN_ImageNet_minimal();
cd ..

clear all;
cd('./CNN');
disp('testing training a new convolutional neural network')
Main_CIFAR_CNN_slow_SGD();
cd ..