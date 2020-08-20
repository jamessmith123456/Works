function [y, Y] = forwordprop(dnn,x)
%UNTITLED3 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
L = size(dnn,2);
m = size(x,2);
Y{1} = x;
for i = 1:L
    z = dnn{i}.W*x + repmat(dnn{i}.b,1,m);
    if dnn{i}.function == "relu"
        y = relu(z);
    end
    if dnn{i}.function == "sigmoid"
        y = sigmoid(z);
    end
    Y{i+1} = y;
    x = y;
end

end

