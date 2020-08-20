%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%����Gabor�˲�����Ӧ����
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function G=create_Gabor()
%�����վ���
G = cell(5,8);
for s = 1:5
    for j = 1:8
        G{s,j}=zeros(32,32);
    end
end
%����40���˲���ģ��
for s = 1:5
    for j = 1:8
%         G{s,9-j} = gabor([32 32],(s-1),j-1,4*pi/5,sqrt(2),2*pi);%�Ժ���Ե���2pi������
        G{s,9-j} = gabor([32 32],(s-1),j-1,4*pi/5,sqrt(2),3*pi/2);
    end
end

%��ʾ40���˲���ģ��
% figure;
% for s = 1:5
%     for j = 1:8        
%        subplot(5,8,(s-1)*8+j);        
%        imshow(real(G{s,j}),[]);
%     end
% end

%��40��ģ����и���Ҷ�任
for s = 1:5
    for j = 1:8
        G{s,j}=fft2(G{s,j});
    end
end

%��ʾ40���˲���ģ�壨����Ҷ�任��
% figure;
% for s = 1:5
%     for j = 1:8        
%         subplot(5,8,(s-1)*8+j);        
%         imshow(real(G{s,j}),[]);
%     end
% end

%����40���˲���ģ������
save mat_Gabor G