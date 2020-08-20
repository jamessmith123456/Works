% 1.用matlab训练一个自己的卷积神经网络中的Alexnet模型，
% 就识别花的种类（数据集的花的种类有十几种，每种80张图片）.
% 不要求训练完的网络最后正确率很高。主要是网络要能跑就行。


% 2.建立深度神经网络独立成分初始化模型，提取神经网络多层次特征，
% 以期获得更加优秀的目标分类和识别神经网络。在训练好的Alexnet模型上，
% 加入ICA的运用（对立成分分析），训练一个新的神经网络，要能跑，同样不要求很高的正确率。网络要能跑。
% 最后要给我一个传统的Alexnet模型，和一个用了ICA的修改过的Alexnet模型。

net = alexnet;
layers = net.Layers(1:end-3);
new_layers = [layers
              fullyConnectedLayer(12,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
              softmaxLayer('name','soft')
              classificationLayer('name','classify')
              ];
imds = imageDatastore('F:\识别花朵\Flowers', 'IncludeSubfolders', true, 'labelsource', 'foldernames');
labelCount = countEachLabel(imds);
[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized'); 
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.0001, ... %雾草这里是个坑！还好有经验
    'MiniBatchSize',3, ...
    'MaxEpochs',40, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsTest, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
% ops = trainingOptions('sgdm', ...
%                       'InitialLearnRate',0.001, ...
%                       'ValidationData',imageTest, ...
%                       'Plots','training-progress', ...
%                       'MiniBatchSize',5, ...
%                       'MaxEpochs',5,...
%                       'ValidationPatience',Inf,...
%                       'Verbose',false);
% 使用训练数据训练网络
gpuDevice(1)
net_train = trainNetwork(imdsTrain,new_layers,options);

% 对验证图像进行分类并计算准确度
YPred = classify(net_train,imdsTest);
YValidation = imdsTest.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);

name = 'E:\识别花朵\Flowers\bluebell\image_0241.jpg';
image_val = imread(name);
% image_val = imresize(imread(name),[227,227]);
[label,conf] = classify(net_train,image_val);

soft = activations(net_train,image_val,'soft');%824464 
YPred = activations(net_train,imdsTest,'soft');
%单独显示第一层卷积的16张图像
act1 = activations(net_train,image_val,'conv1');%824464 
sz = size(act1);
act1 = reshape(act1,[227 227 1 16]);
imshow(I)
%整体显示第一层卷积的16张图像
act1ch32 = act1(:,:,:,8);
act1ch32 = mat2gray(act1ch32);
act1ch32 = imresize(act1ch32,[227,227]);
% I = imtile({im,act1ch32});
imshow(act1ch32)

%单独显示第二层卷积的图像
act2 = activations(net,image_val,'conv2');
sz = size(act2);
act2 = reshape(act2,[sz(1) sz(2) 1 sz(3)]);
I = imtile(mat2gray(act2),'GridSize',[8 12]);
imshow(I)
%整体显示第二层卷积的图像
act1ch32 = act1(:,:,:,32);
act1ch32 = mat2gray(act1ch32);
act1ch32 = imresize(act1ch32,imgSize);
I = imtile({im,act1ch32});
imshow(I)

%单独显示第三层卷积中的每幅图像
act3 = activations(net,image_val,'conv3');
sz = size(act3);
act1 = reshape(act3,[sz(1) sz(2) 1 sz(3)]);
I = imtile(mat2gray(act3),'GridSize',[8 12]);
imshow(I)
%整体显示第三层卷积的图像
act1ch32 = act3(:,:,:,32);
act1ch32 = mat2gray(act1ch32);
act1ch32 = imresize(act1ch32,imgSize);
I = imtile({im,act1ch32});
imshow(I)