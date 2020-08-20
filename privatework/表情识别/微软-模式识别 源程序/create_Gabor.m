%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%创建Gabor滤波器响应矩阵
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function G=create_Gabor()
%创建空矩阵
G = cell(5,8);
for s = 1:5
    for j = 1:8
        G{s,j}=zeros(32,32);
    end
end
%创建40个滤波器模板
for s = 1:5
    for j = 1:8
%         G{s,9-j} = gabor([32 32],(s-1),j-1,4*pi/5,sqrt(2),2*pi);%以后可以调到2pi再试试
        G{s,9-j} = gabor([32 32],(s-1),j-1,4*pi/5,sqrt(2),3*pi/2);
    end
end

%显示40个滤波器模板
% figure;
% for s = 1:5
%     for j = 1:8        
%        subplot(5,8,(s-1)*8+j);        
%        imshow(real(G{s,j}),[]);
%     end
% end

%对40个模板进行傅里叶变换
for s = 1:5
    for j = 1:8
        G{s,j}=fft2(G{s,j});
    end
end

%显示40个滤波器模板（傅里叶变换后）
% figure;
% for s = 1:5
%     for j = 1:8        
%         subplot(5,8,(s-1)*8+j);        
%         imshow(real(G{s,j}),[]);
%     end
% end

%保存40个滤波器模板数据
save mat_Gabor G