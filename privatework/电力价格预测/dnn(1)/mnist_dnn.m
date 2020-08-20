clear all
load('mnist_uint8.mat');
test_x = (double(test_x)/255)';
train_x = (double(train_x)/255)';
test_y = double(test_y.');
train_y = double(train_y.');
K.f = {"relu","relu","relu","sigmoid"};
K.a = [784,400,300,500,10];
[net,P] = creatnn(K);
P.method = "RMSProp";
P.learning_rate = 0.001;
m = size(train_x,2);
batch_size = 100;
MAX_P = 2000;
global E;
for i = 1:MAX_P
    q = randi(m,1,batch_size);
    train = train_x(:,q);
    label = train_y(:,q);
    net = backprop(train,label,net,P);
    if mod(i,50) == 0
        [output,~] = forwordprop(net,train);
        [~,index0] = max(output);
        [~,index1] = max(label);
        rate = sum(index0 == index1)/batch_size;
        fprintf("第%d训练包的正确率:%f\n",i,rate)
        [output,~] = forwordprop(net,test_x);
        [~,index0] = max(output);
        [~,index1] = max(test_y);
        rate = sum(index0 == index1)/size(test_x,2);
        fprintf("测试集的正确率:%f\n",rate)
    end
end