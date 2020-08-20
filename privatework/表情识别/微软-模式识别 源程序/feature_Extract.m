%Gabor������ȡ����
%����˵����
%image       : �����ͼ�����һ�������ݾ���
%width,height: ͼƬ�ߴ�

function featVector = feature_Extract(image,G,width,height)

%����Gabor�˲�������������ȡ
Temp = cell(5,8);%���������ݴ����
for s = 1:5 %5���߶�
    for j = 1:8 %8������
        %��ʾ����FFT����Ƶ�����г˻����ٷ��任���������õ����ǿ���ġ�
        Temp{s,j} = ifft2(G{s,j}.*fft2(double(image),32,32),height,width);
    end
end

%ǿ�Ƹ���ȥ�����ݣ���Сά����ԭ����1/4,�����Զ�ѹ��
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