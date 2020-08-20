% 1.��matlabѵ��һ���Լ��ľ���������е�Alexnetģ�ͣ�
% ��ʶ�𻨵����ࣨ���ݼ��Ļ���������ʮ���֣�ÿ��80��ͼƬ��.
% ��Ҫ��ѵ��������������ȷ�ʺܸߡ���Ҫ������Ҫ���ܾ��С�


% 2.�����������������ɷֳ�ʼ��ģ�ͣ���ȡ���������������
% ���ڻ�ø��������Ŀ������ʶ�������硣��ѵ���õ�Alexnetģ���ϣ�
% ����ICA�����ã������ɷַ�������ѵ��һ���µ������磬Ҫ���ܣ�ͬ����Ҫ��ܸߵ���ȷ�ʡ�����Ҫ���ܡ�
% ���Ҫ����һ����ͳ��Alexnetģ�ͣ���һ������ICA���޸Ĺ���Alexnetģ�͡�

net = alexnet;
layers = net.Layers(1:end-3);
new_layers = [layers
              fullyConnectedLayer(12,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
              softmaxLayer('name','soft')
              classificationLayer('name','classify')
              ];
imds = imageDatastore('F:\ʶ�𻨶�\Flowers', 'IncludeSubfolders', true, 'labelsource', 'foldernames');
labelCount = countEachLabel(imds);
[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized'); 
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.0001, ... %��������Ǹ��ӣ������о���
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
% ʹ��ѵ������ѵ������
gpuDevice(1)
net_train = trainNetwork(imdsTrain,new_layers,options);

% ����֤ͼ����з��ಢ����׼ȷ��
YPred = classify(net_train,imdsTest);
YValidation = imdsTest.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);

name = 'E:\ʶ�𻨶�\Flowers\bluebell\image_0241.jpg';
image_val = imread(name);
% image_val = imresize(imread(name),[227,227]);
[label,conf] = classify(net_train,image_val);

soft = activations(net_train,image_val,'soft');%824464 
YPred = activations(net_train,imdsTest,'soft');
%������ʾ��һ������16��ͼ��
act1 = activations(net_train,image_val,'conv1');%824464 
sz = size(act1);
act1 = reshape(act1,[227 227 1 16]);
imshow(I)
%������ʾ��һ������16��ͼ��
act1ch32 = act1(:,:,:,8);
act1ch32 = mat2gray(act1ch32);
act1ch32 = imresize(act1ch32,[227,227]);
% I = imtile({im,act1ch32});
imshow(act1ch32)

%������ʾ�ڶ�������ͼ��
act2 = activations(net,image_val,'conv2');
sz = size(act2);
act2 = reshape(act2,[sz(1) sz(2) 1 sz(3)]);
I = imtile(mat2gray(act2),'GridSize',[8 12]);
imshow(I)
%������ʾ�ڶ�������ͼ��
act1ch32 = act1(:,:,:,32);
act1ch32 = mat2gray(act1ch32);
act1ch32 = imresize(act1ch32,imgSize);
I = imtile({im,act1ch32});
imshow(I)

%������ʾ���������е�ÿ��ͼ��
act3 = activations(net,image_val,'conv3');
sz = size(act3);
act1 = reshape(act3,[sz(1) sz(2) 1 sz(3)]);
I = imtile(mat2gray(act3),'GridSize',[8 12]);
imshow(I)
%������ʾ����������ͼ��
act1ch32 = act3(:,:,:,32);
act1ch32 = mat2gray(act1ch32);
act1ch32 = imresize(act1ch32,imgSize);
I = imtile({im,act1ch32});
imshow(I)