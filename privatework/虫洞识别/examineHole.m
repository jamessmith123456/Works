
% %下面开始自己的算法步骤
% path='E:\虫洞识别\粗匹配';
% files = dir(path);
% 
% i=3:147
% i=3
% img_path = [path,'\',files(i).name];
% img = rgb2gray(imread(img_path));
% % imgdouble = double(img);
% % [m,n] = size(imgdouble);
% % % tidu = [];
% % % for i=1:n-1
% % %     temp = 0;
% % %     for j=1:m
% % %         temp = temp+abs(imgdouble(j,i+1)-imgdouble(j,i));
% % %     end
% % %     if i==1
% % %         tidu = temp;
% % %     else
% % %         tidu = [tidu;temp];
% % %     end
% % % end
% % % [A,B] = sort(tidu);
% % tidu = zeros(m-1,n-1);
% % pos = {};
% % for i=2:m
% %     for j=2:n
% %         tidu(i,j) = (imgdouble(i,j)-imgdouble(i-1,j))^2+(imgdouble(i,j)-imgdouble(i,j-1))^2;
% %         pos{i,j} = [i,j];
% %     end
% % end
% % 
% % intevarl = 20;
% % [mm,nn] = size(tidu);
% % groundtidu = zeros(mm,nn);
% % for i=intevarl+1:mm-intevarl
% %     for j=intevarl+1:nn-intevarl
% %         temp = 0;
% %         for k=1:intevarl
% %             for kk=1:intevarl
% %                 temp = temp +tidu(i-k,j-kk);
% %             end
% %         end
% %         groundtidu(i,j)=temp;
% %     end
% % end
% % 
% % a = reshape(groundtidu,[1,mm*nn]);
% % aa = sort(a);
% % %83549 83387 83020 82713 82647 82562 82404 82279 82251
% % [p1,p2] = find(groundtidu==83549);%863,914
% % [p1,p2] = find(groundtidu==83387);% 864 914
% % [p1,p2] = find(groundtidu==83020); % 862 914
% % [p1,p2] = find(groundtidu==82713);% 865 914
% % [p1,p2] = find(groundtidu==82647);%863 915
% % [p1,p2] = find(groundtidu==82562);%862 915




%上面经过试验都不行，下面这个方法应该可行？我的天鸭终于成功了.....
path='E:\虫洞识别\桌面\边缘提取';
files = dir(path);
ABpos = {};
names = {};
for i=3:147
    img_path = [path,'\',files(i).name];
    % img = imread(img_path);
    img = rgb2gray(imread(img_path));
    imgdouble = double(img);
    %针对每一张图片还是要找出它的上边缘或者右边缘
    rowsum = sum(imgdouble);
    colsum = sum(imgdouble,2);
    [rowA,rowB] = sort(rowsum);
    [colA,colB] = sort(colsum);

    imshow(img);% display Intensity Image
    hold on; 
    plot(rowB(end),colB(end) ,'g*');%911 59
    hold on;
    %
    imgdouble2 = imgdouble(colB(end)-20:colB(end)+20,1:100);
    img2 = img(colB(end)-20:colB(end)+20,1:100);
    rowsum = sum(imgdouble2);
    [rowA2,rowB2] = sort(rowsum);
    plot(rowB2(end),colB(end) ,'r*');
    pause(0.5);
    ABpos{i}=[rowB2(end),colB(end);rowB(end),colB(end)];
    names{i}= files(i).name;
end


clear;
%下面的思路就是根据上面得到的AB点位置信息,截取出每一个原图的AB为顶点的区域，然后相对于图片1，进行缩放
%也就是把所有的图片,都按照第一张图片进行缩放
path='E:\虫洞识别\粗匹配';
files = dir(path);
i=3;
img_path = [path,'\',files(i).name];
img = rgb2gray(imread(img_path));
StandardLength = ABpos{3}(2,1)-ABpos{3}(1,1);
img2 = img(ABpos{3}(1,2):ABpos{3}(1,2)+StandardLength,ABpos{3}(1,1):ABpos{3}(2,1));
pathSave = 'E:\虫洞识别\归一化\';
imwrite(img2,[pathSave,names{3}]);
for i=4:size(files,1)
    img_path = [path,'\',files(i).name];
    img = rgb2gray(imread(img_path));
    pwdLength = ABpos{i}(2,1)-ABpos{i}(1,1);
    if i==55
        continue;
    else
        if (ABpos{i}(1,2)+pwdLength)<size(img,2)
            img2 = img(ABpos{i}(1,2):ABpos{i}(1,2)+pwdLength,ABpos{i}(1,1):ABpos{i}(2,1));
        else
            img2 = img(ABpos{i}(1,2)-pwdLength:ABpos{i}(1,2),ABpos{i}(1,1):ABpos{i}(2,1));
        end
    end
    F = imresize(img2,[StandardLength,StandardLength]);
    imwrite(F,[pathSave,names{i}]);
end

%     MATLAB诡异的彩蛋？？？
%     bw = im2bw( image,0.15);%转二值图
%     imshow(bw);
%完美，上面一步已经将所有的图片全部裁剪缩放并且存储到E:\虫洞识别\归一化\这个路径下了
%现在就是检测每一张图里的裂缝了
clear;
path = 'E:\虫洞识别\归一化';
files = dir(path);
AllBlackPos = {};
pathLeakSave = 'E:\虫洞识别\裂缝\'
for i=3:size(files,1)
    img_path = [path,'\',files(i).name];
    img = imread(img_path); 
    bw = im2bw( img,0.4);%转二值图
    [tempA,tempB] = find(bw==0); %tempA表示0元素对应的行 tempB表示0元素对应的列
    recordpos = [];
    if isempty(tempA)==0
        for j=1:size(tempA,1)
            x=tempA(j);
            y=tempB(j);
            if x-1<=0 || x+1>=size(bw,1) || y-1<=0 || y+1>=size(bw,1)
                    if j==1
                        recordpos = [x,y];
                    else
                        recordpos = [recordpos;x,y];
                    end
            else if bw(x-1,y)==1 || bw(x+1,y)==1 || bw(x,y-1)==1 || bw(x,y+1)==1
                    if j==1
                        recordpos = [x,y];
                    else
                        recordpos = [recordpos;x,y];
                    end
                end
            end
        end
        imshow(bw);
        hold on;
        scatter(recordpos(:,2),recordpos(:,1),'r');
        pause(0.5);
    end
%     AllBlackPos{i}=[tempA,tempB];
    AllBlackPos{i}=recordpos;
    length = size(files(i).name,2);
    if length==21
        tempname = files(i).name(16);
    else if length==22
            tempname = files(i).name(16:17);
        else if length==23
                tempname = files(i).name(16:18);
            end
        end
    end
    imwrite(bw,[pathLeakSave,tempname,'.png']);
end

% %先做实验,发现0.4最好
% % img_path = 'E:\虫洞识别\归一化\橡20180119-0.1 (86).jpg';
% img_path = 'E:\虫洞识别\归一化\橡20180119-0.1 (50).jpg';
% img = imread(img_path);
% subplot(1,4,1);
% imshow(img);
% bw = im2bw( img,0.3);
% subplot(1,4,2);
% imshow(bw);
% bw2 = im2bw( img,0.4);
% subplot(1,4,3);
% imshow(bw2);
% bw3 = im2bw( img,0.45);
% subplot(1,4,4);
% imshow(bw3);




%上面得到的AllBlackPos里面存储的就是每张图像的可能为虫洞信息的黑色像素点位置
%需要进行过滤,把误认为是虫洞的部分给删除掉,不过滤也可以，画的三维度有点毛糙
%下面是根据上面探测得到的虫洞区域,进行三维图像的绘制
%绘制时发现一个问题,上面提取出的核的区域，最好还是求一下边缘，这样可以减少画图时候的工作量
Z=[];
X=[];
Y=[];
t=1;
for i=3:size(AllBlackPos,2)
    temp = AllBlackPos(i);
    if isempty(temp{1})==0
        if i==3
            X = temp{1}(:,1);
            Y = temp{1}(:,2);
            Z = t*ones(size(temp{1},1),1);
        else
            X = [X;temp{1}(:,1)];
            Y = [Y;temp{1}(:,2)];
            Z = [Z;t*ones(size(temp{1},1),1)];
        end
        t=t+1;
    end
%     temp = AllBlackPos(3);
%     X = temp{1}(:,1);
%     Y = temp{1}(:,2);
%     Z = t*ones(size(temp{1},1),size(temp{1},1));
%     surf(X,Y,Z);
%     hold on;
%     t=t+1;
end
scatter3(X,Y,Z,'.');