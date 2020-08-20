function [y] = relu(x)
%UNTITLED4 此处显示有关此函数的摘要
%   此处显示详细说明
p = (x > 0);
y = x.*p;
end

