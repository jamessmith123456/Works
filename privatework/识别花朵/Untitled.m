�о��˳��õ����ֱ�Ե����㷨��Roberts���ӡ�Prewitt���ӡ�Sobel���ӡ� Kirsch���ӡ�Canny����

%����ͼ���Ե���
clc; close all
fid = fopen('200kv0.5ua��1.dr','r');
fseek(fid, 1024, 'bof'); % ����ͷ�ļ��ֽ�
proj = fread(fid, [2048, 2048], 'float');
fclose(fid);

figure();
imshow(proj, []);

proj1 = proj(40:2008, 40:2008); % % ѡ��ͼ���е�һ����
figure();
imshow(proj1, []);
