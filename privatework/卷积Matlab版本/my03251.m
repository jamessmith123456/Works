%���Ȼ�ȡ����
all_data = zeros(32,32,3,60000);
all_labels = zeros(10,60000);
all_category = {'airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'};
path = './cifar/';
num =1;
for i=1:10
    temppath = [path,all_category{i}];
    all_file = dir(temppath);
    for j=1:6000
        tempimg = [temppath,'/',all_file(j+2).name];
        img = imread(tempimg);
        all_data(:,:,:,num) = im2double(img);
        all_labels(j,num) = 1;
        num = num+1;
    end
end

randIndex = randperm(size(all_data,4));
all_data = all_data(:,:,:,randIndex);
all_labels = all_labels(:,randIndex);

train_length = round(size(all_data,4)*0.8);
cnn.layers = {
    struct('type', 'i')  % �����
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5)  % �����
    struct('type', 's', 'scale', 2)  % �ϲ���
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) % �����
    struct('type', 's', 'scale', 2) % �ϲ���
    
};

cnn = cnnsetup(cnn, train_x, train_y);

opts.alpha = 1;
opts.batchsize = 50;
opts.numepochs = 1;
cnn = cnntrain(cnn, train_x, train_y, opts);
[er, bad] = cnntest(cnn, test_x, test_y);

train_x = double(reshape(train_x',28,28,60000))/255;
test_x = double(reshape(test_x',28,28,10000))/255;
train_y = double(train_y');
test_y = double(test_y');

% ��һ��ѭ����Ҫ200�룬һ��epoch���Ի��11%����
% 100 ��epochs ֮����Ի��1.2%����
rand('state',0)
% ����ṹ
cnn.layers = {
    struct('type', 'i')  % �����
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5)  % �����
    struct('type', 's', 'scale', 2)  % �ϲ���
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) % �����
    struct('type', 's', 'scale', 2) % �ϲ���
    
};
% �����ʼ��
cnn = cnnsetup(cnn, train_x, train_y);

% ����
opts.alpha = 1;
opts.batchsize = 50;
opts.numepochs = 1;

% ѵ��
cnn = cnntrain(cnn, train_x, train_y, opts);

% ��֤���
[er, bad] = cnntest(cnn, test_x, test_y);
pre = cnnpre(cnn,test_x);
% ��ӡ�������
figure; plot(cnn.rL);

% ���er>=0.12 �򱨴�
assert(er<0.12, 'Too big error');