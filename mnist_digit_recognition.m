clc;
clear;
close all;

% Load MNIST data
[XTrain, YTrain] = digitTrain4DArrayData;
[XTest, YTest] = digitTest4DArrayData;

% Visualize sample images
figure;
for i = 1:20
    subplot(4, 5, i);
    imshow(XTrain(:,:,:,i));
    title(['Label: ', char(YTrain(i))]);
end

% Define CNN architecture
layers = [
    imageInputLayer([28 28 1])

    convolution2dLayer(3, 8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
];

% Training options
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 5, ...
    'MiniBatchSize', 128, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% Train the network
net = trainNetwork(XTrain, YTrain, layers, options);

% Predict and evaluate
YPred = classify(net, XTest);
accuracy = sum(YPred == YTest)/numel(YTest);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

% Display predictions
idx = randperm(numel(YTest), 16);
figure;
for i = 1:16
    subplot(4,4,i);
    imshow(XTest(:,:,:,idx(i)));
    title(['Pred: ', char(YPred(idx(i)))]);
end
