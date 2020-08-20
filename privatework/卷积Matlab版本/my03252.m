imds = imageDatastore('./cifar', 'IncludeSubfolders', true, 'labelsource', 'foldernames');
labelCount = countEachLabel(imds);
[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized'); 
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ... %雾草这里是个坑！还好有经验
    'MiniBatchSize',3, ...
    'MaxEpochs',10, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsTest, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

new_layers = [imageInputLayer([32 32 3])
          convolution2dLayer(5,20)
          reluLayer
          maxPooling2dLayer(2,'Stride',2)
          fullyConnectedLayer(10)
          softmaxLayer
          classificationLayer];
net_train = trainNetwork(imdsTrain,new_layers,options);

YPred = classify(net_train,imdsTest);
YValidation = imdsTest.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);