研究了常用的五种边缘检测算法：Roberts算子、Prewitt算子、Sobel算子、 Kirsch算子、Canny算子

%刀具图像边缘检测
clc; close all
fid = fopen('200kv0.5ua大刀1.dr','r');
fseek(fid, 1024, 'bof'); % 跳过头文件字节
proj = fread(fid, [2048, 2048], 'float');
fclose(fid);

figure();
imshow(proj, []);

proj1 = proj(40:2008, 40:2008); % % 选择图像中的一部分
figure();
imshow(proj1, []);
