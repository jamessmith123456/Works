clear all;
sample = rgb2gray(imread(img_path));
mean3Sample = filter2(fspecial('average',3),sample)/255;
mean5Sample = filter2(fspecial('average',5),sample)/255;
mean7Sample = filter2(fspecial('average',7),sample)/255;
gaussianSample = filter2(fspecial('gaussian'),sample)/255;

subplot(2,2,1);
imshow(sample); %ԭʼͼ��
subplot(2,2,2);
imshow(mean7Sample); %���þ�ֵ����ƽ������
subplot(2,2,3);
imshow(sample); %ԭʼͼ��
subplot(2,2,4);
imshow(gaussianSample); %��˹�˲�������ƽ������

%���á�prewitt������:
prewittSample = uint8(filter2(fspecial('prewitt'),sample));
imshow(prewittSample);
%���á� sobel������:
sobelSample = uint8(filter2(fspecial('sobel'),sample));
imshow(sobelSample);


%���á�ԭͼ*2-ƽ��ͼ�񡱷���:
subSample = sample.*2 - uint8(mean7Sample);
imshow(subSample);
%���á�ԭͼ+��Ե����ͼ�񡱷���
addSample = sample + uint8(prewittSample);
imshow(addSample);