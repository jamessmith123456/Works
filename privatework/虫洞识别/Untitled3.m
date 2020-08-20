clear all;
sample = rgb2gray(imread(img_path));
mean3Sample = filter2(fspecial('average',3),sample)/255;
mean5Sample = filter2(fspecial('average',5),sample)/255;
mean7Sample = filter2(fspecial('average',7),sample)/255;
gaussianSample = filter2(fspecial('gaussian'),sample)/255;

subplot(2,2,1);
imshow(sample); %原始图像
subplot(2,2,2);
imshow(mean7Sample); %采用均值进行平滑处理
subplot(2,2,3);
imshow(sample); %原始图像
subplot(2,2,4);
imshow(gaussianSample); %高斯滤波器进行平滑处理

%采用’prewitt’算子:
prewittSample = uint8(filter2(fspecial('prewitt'),sample));
imshow(prewittSample);
%采用’ sobel’算子:
sobelSample = uint8(filter2(fspecial('sobel'),sample));
imshow(sobelSample);


%采用“原图*2-平滑图像”方法:
subSample = sample.*2 - uint8(mean7Sample);
imshow(subSample);
%采用“原图+边缘处理图像”方法
addSample = sample + uint8(prewittSample);
imshow(addSample);