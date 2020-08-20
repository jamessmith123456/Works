function [dnn] = backprop(x,label,dnn,parameter)
%UNTITLED2 此处显示有关此函数的摘要
%   parameter 是结构体，包括参数：
%                       learning_rate: 学习率
%                       momentum： 动量系数,一般为0.5，0.9，0.99
%                       attenuation_rate： 衰减系数
%                       delta：稳定数值
%                       step: 步长 一般为 0.001
%                       method: 方法{'SGD','mSGD','nSGD','AdaGrad','RMSProp','nRMSProp','Adam'}
%
L = size(dnn,2)+1;
m = size(x,2);
[y, Y] = forwordprop(dnn,x);
g = -label./y + (1 - label)./(1 - y);
method = {"SGD","mSGD","nSGD","AdaGrad","RMSProp","nRMSProp","Adam"};

persistent global_step;
if isempty(global_step)
   global_step = 0;
end
global_step = global_step + 1;
% fprintf("global_step %d\n",global_step)
global E;
E(global_step) = sum(sum(-label.*log(y)-(1 - label).*log(1 - y)))/m;
persistent V;
if isempty(V)
    for i = 1:L-1
        V{i}.vw = dnn{i}.W*0;
        V{i}.vb = dnn{i}.b*0;
    end
end

if parameter.method == method{1,1}
    for i = L : -1 : 2
        if dnn{i-1}.function == "relu"
            g = g.*(Y{i} > 0);
        end
        if dnn{i-1}.function == "sigmoid"
            g = g.*Y{i}.*(1 - Y{i});
        end
            dw = g*Y{i - 1}.'/m;
            db = sum(g,2)/m;
            g = dnn{i-1}.W'*g;
            dnn{i-1}.W = dnn{i-1}.W - parameter.learning_rate*dw;
            dnn{i-1}.b = dnn{i-1}.b - parameter.learning_rate*db;
    end                                                                                                                                                                                                                                                                                        
end

if parameter.method == method{1,2}
    for i = L : -1 : 2
        if dnn{i-1}.function == "relu"
            g = g.*(Y{i} > 0);
        end
        if dnn{i-1}.function == "sigmoid"
            g = g.*Y{i}.*(1 - Y{i});
        end
            dw = g*Y{i - 1}.'/m;
            db = sum(g,2)/m;
            g = dnn{i-1}.W'*g;
            V{i-1}.vw = parameter.momentum*V{i-1}.vw - parameter.learning_rate*dw; 
            V{i-1}.vb = parameter.momentum*V{i-1}.vb - parameter.learning_rate*db;
            dnn{i-1}.W = dnn{i-1}.W + V{i-1}.vw;
            dnn{i-1}.b = dnn{i-1}.b + V{i-1}.vb;
    end
end

if parameter.method == method{1,3} % 未实现    
    for i = L : -1 : 2
        if dnn{i-1}.function == "relu"
            g = g.*(Y{i} > 0);
        end
        if dnn{i-1}.function == "sigmoid"
            g = g.*Y{i}.*(1 - Y{i});
        end
            dw = g*Y{i - 1}.'/m;
            db = sum(g,2)/m;
            g = dnn{i-1}.W'*g;
            V{i-1}.vw = parameter.momentum*V{i-1}.vw - parameter.learning_rate*dw; 
            V{i-1}.vb = parameter.momentum*V{i-1}.vb - parameter.learning_rate*db;
            dnn{i-1}.W = dnn{i-1}.W + V{i-1}.vw;
            dnn{i-1}.b = dnn{i-1}.b + V{i-1}.vb;
    end
end

if parameter.method == method{1,4}     
    for i = L : -1 : 2
        if dnn{i-1}.function == "relu"
            g = g.*(Y{i} > 0);
        end
        if dnn{i-1}.function == "sigmoid"
            g = g.*Y{i}.*(1 - Y{i});
        end
            dw = g*Y{i - 1}.'/m;
            db = sum(g,2)/m;
            g = dnn{i-1}.W'*g;
            V{i-1}.vw = V{i-1}.vw + dw.*dw; 
            V{i-1}.vb = V{i-1}.vb + db.*db;
            dnn{i-1}.W = dnn{i-1}.W - parameter.learning_rate./(parameter.delta + sqrt(V{i-1}.vw)).*dw;
            dnn{i-1}.b = dnn{i-1}.b - parameter.learning_rate./(parameter.delta + sqrt(V{i-1}.vb)).*db;
    end
end

if parameter.method == method{1,5}     
    for i = L : -1 : 2
        if dnn{i-1}.function == "relu"
            g = g.*(Y{i} > 0);
        end
        if dnn{i-1}.function == "sigmoid"
            g = g.*Y{i}.*(1 - Y{i});
        end
            dw = g*Y{i - 1}.'/m;
            db = sum(g,2)/m;
            g = dnn{i-1}.W'*g;
            V{i-1}.vw = parameter.attenuation_rate*V{i-1}.vw + (1 - parameter.attenuation_rate)*dw.*dw; 
            V{i-1}.vb = parameter.attenuation_rate*V{i-1}.vb + (1 - parameter.attenuation_rate)*db.*db;
            dnn{i-1}.W = dnn{i-1}.W - parameter.learning_rate./sqrt(parameter.delta + V{i-1}.vw).*dw;
            dnn{i-1}.b = dnn{i-1}.b - parameter.learning_rate./sqrt(parameter.delta + V{i-1}.vb).*db;
    end
end

persistent s;
if parameter.method == method{1,7}  
    if isempty(s)
        for i = 1:L-1
            s{i}.vw = dnn{i}.W*0;
            s{i}.vb = dnn{i}.b*0;
        end
    end
    for i = L : -1 : 2
        if dnn{i-1}.function == "relu"
            g = g.*(Y{i} > 0);
        end
        if dnn{i-1}.function == "sigmoid"
            g = g.*Y{i}.*(1 - Y{i});
        end
            dw = g*Y{i - 1}.'/m;
            db = sum(g,2)/m;
            g = dnn{i-1}.W'*g;
            s{i-1}.vw = parameter.beta2*s{i-1}.vw + (1 - parameter.beta1)*dw; 
            s{i-1}.vb = parameter.beta2*s{i-1}.vb + (1 - parameter.beta1)*db;
            V{i-1}.vw = parameter.beta2*V{i-1}.vw + (1 - parameter.beta2)*dw.*dw; 
            V{i-1}.vb = parameter.beta2*V{i-1}.vb + (1 - parameter.beta2)*db.*db;
            
            dnn{i-1}.W = dnn{i-1}.W - parameter.learning_rate*(s{i-1}.vw/(1-parameter.beta1.^global_step))./(parameter.delta + sqrt(V{i-1}.vw./(1 - parameter.beta2.^global_step)));
            dnn{i-1}.b = dnn{i-1}.b - parameter.learning_rate*(s{i-1}.vb/(1-parameter.beta1.^global_step))./(parameter.delta + sqrt(V{i-1}.vb./(1 - parameter.beta2.^global_step)));
    end
end
end

