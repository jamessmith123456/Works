%为经过ICA独立性分析的特征单独构建一个网络,因为此时每个样本的特征是二维的，1*1000
%且其含义本身就代表经过卷积操作得到的特征，所以没必要再添加卷积层了
%直接添加全连接层，也就是构建普通的神经网络进行分类即可
inputSize = [1 1000 1];
numClasses = 12;

imds2 = imageDatastore('F:\识别花朵\ICA_NewFeatures', 'IncludeSubfolders', true, 'labelsource', 'foldernames');
labelCount = countEachLabel(imds2);
[imdsTrain2,imdsTest2] = splitEachLabel(imds2,0.7,'randomized'); 
layers = [
    imageInputLayer(inputSize)
    fullyConnectedLayer(500)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
options = trainingOptions('sgdm', ...
    'MaxEpochs',4, ...
    'ValidationData',imdsTest2, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
net_train2 = trainNetwork(imdsTrain2,layers,options);
YPred = classify(net_train2,imdsTest2);
YValidation = imdsTest2.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);