%首先获取数据
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
    struct('type', 'i')  % 输入层
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5)  % 卷积层
    struct('type', 's', 'scale', 2)  % 上采样
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) % 卷积层
    struct('type', 's', 'scale', 2) % 上采样
    
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

% 跑一次循环需要200秒，一个epoch可以获得11%的误差；
% 100 个epochs 之后可以获得1.2%的误差。
rand('state',0)
% 网络结构
cnn.layers = {
    struct('type', 'i')  % 输入层
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5)  % 卷积层
    struct('type', 's', 'scale', 2)  % 上采样
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) % 卷积层
    struct('type', 's', 'scale', 2) % 上采样
    
};
% 网络初始化
cnn = cnnsetup(cnn, train_x, train_y);

% 参数
opts.alpha = 1;
opts.batchsize = 50;
opts.numepochs = 1;

% 训练
cnn = cnntrain(cnn, train_x, train_y, opts);

% 验证误差
[er, bad] = cnntest(cnn, test_x, test_y);
pre = cnnpre(cnn,test_x);
% 打印均方误差
figure; plot(cnn.rL);

% 如果er>=0.12 则报错
assert(er<0.12, 'Too big error');