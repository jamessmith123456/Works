%Gabor特征提取函数
%参数说明：
%image       : 输入的图像矩阵，一定是数据矩阵
%width,height: 图片尺寸

function featVector = feature_Extract(image,G,width,height)

%利用Gabor滤波器进行特征提取
Temp = cell(5,8);%创建特征暂存矩阵
for s = 1:5 %5个尺度
    for j = 1:8 %8个方向
        %显示进行FFT，到频域后进行乘积，再反变换回来，最后得到的是空域的。
        Temp{s,j} = ifft2(G{s,j}.*fft2(double(image),32,32),height,width);
    end
end

%强制隔行去掉数据，减小维数到原来的1/4,空行自动压缩
Temp = abs(cell2mat(Temp));
% [m n]=size(Temp);
% for i=1:m
%     if mod(i,4)==2||mod(i,4)==3||mod(i,4)==0
%         Temp(i,:)=[];
%     end
% end
% for i=1:n
%     if mod(i,4)==2||mod(i,4)==3||mod(i,4)==0
%         Temp(:,i)=[];
%     end
% end

% % Temp (3:3:end,:)=[];
Temp (2:2:end,:)=[];
Temp (2:2:end,:)=[];
Temp (2:2:end,:)=[];
% % Temp (:,3:3:end)=[];
Temp (:,2:2:end)=[];
Temp (:,2:2:end)=[];
Temp (:,2:2:end)=[];
Temp = premnmx(Temp);
[m n]=size(Temp);
featVector = reshape (Temp,[1 m*n]);