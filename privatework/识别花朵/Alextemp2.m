%Ϊ����ICA�����Է�����������������һ������,��Ϊ��ʱÿ�������������Ƕ�ά�ģ�1*1000
%���京�屾��ʹ�������������õ�������������û��Ҫ����Ӿ������
%ֱ�����ȫ���Ӳ㣬Ҳ���ǹ�����ͨ����������з��༴��
inputSize = [1 1000 1];
numClasses = 12;

imds2 = imageDatastore('F:\ʶ�𻨶�\ICA_NewFeatures', 'IncludeSubfolders', true, 'labelsource', 'foldernames');
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