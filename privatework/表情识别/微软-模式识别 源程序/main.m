%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%main函数
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
clc;
close all;
warning off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%载入数据MAT文件
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%//////////////////////////////////////////////////////////////////////////
%加载样本信息矩阵，这是最原始的矩阵，是手动录入数据形成，必须存在
load('mat_SamplesInfo.mat'); %里面包含Samples矩阵，213*5
fprintf ('<加载> 原始样本信息矩阵...\n\n');

%//////////////////////////////////////////////////////////////////////////
%样本数据矩阵，该矩阵存放着每张训练人脸的3个ROI和全脸数据
if ~exist('mat_SamplesData.mat','file') %samples_Data，213*12的cell
    fprintf ('<创建> 样本数据矩阵...\n\n');
    samples_Data=create_samplesData('JAFFE_ROI/','JAFFE_CUT/',Samples);%创建该矩阵
else
    load('mat_SamplesData.mat');
    fprintf ('<加载> 样本数据矩阵...\n\n');
end

%//////////////////////////////////////////////////////////////////////////
%加载Gabor滤波器响应矩阵
if ~exist('mat_Gabor.mat','file') %G矩阵，5*8的cell，每个cell里32*32数据
    fprintf ('<创建> Gabor滤波器响应矩阵...\n\n');
    G=create_Gabor();%创建该矩阵
else
    load('mat_Gabor.mat');
    fprintf ('<加载> Gabor滤波器响应矩阵...\n\n');
end

%//////////////////////////////////////////////////////////////////////////
%创建Gabor特征训练数据集
% if ~exist('mat_trainSet.mat','file')||~exist('mat_trainSetP.mat','file')||~exist('mat_trainSetT.mat','file')
if ~exist('mat_trainSetP.mat','file')||~exist('mat_trainSetT.mat','file')    
    fprintf ('<创建> Gabor特征训练数据集...\n\n');
    [trainSet,P,T]=create_trainSet(G,samples_Data);
else
    askStr=input('是否重新创建训练数据集？(Y/N) :','s');
    if askStr=='Y'||askStr=='y'
        [trainSet,P,T]=create_trainSet(G,samples_Data);
    else
        load('mat_trainSetP.mat'); %P 354*213的double矩阵    354是特征维度？7是表情个数？
        load('mat_trainSetT.mat'); %T 7*213的double矩阵
    end
    trainP=P;
    trainT=T;
    testP=P; %这里啥意思？
    testT=T;
    %这里是归一化函数，maxP minP分别是P中最大值、最小值   maxT minT分别是T中最大值、最小值 
    %PP是354*213，归一化之后的P  TT是归一化之后的T,7*213 
    [PP,minP,maxP,TT,minT,maxT]=premnmx(P,T);%获得最大值、最小值  
    fprintf ('<加载> Gabor特征训练数据集...\n\n');
end

%//////////////////////////////////////////////////////////////////////////
%加载神经网络分类器
if exist('mat_net.mat','file')
    fprintf ('<加载> 已有神经网络分类器\n\n');
    load('mat_net.mat'); %Net，已经有的神经网络分类器
else
    fprintf ('<提示> 没有找到神经网络分类器，请重新训练\n\n');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%主菜单循环
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while (1==1)
    choice=menu('        表情识别           ',...
                '>>JAFFE自测试<<',...
                '>>训练BP神经网络分类器<<',...
                '>>测试表情图片<<',...
                '>>手动人脸区域剪切<<',...
                '>>手动ROI区域(眼睛 鼻子 嘴巴)剪切<<',...
                '>>KNN聚类分类<<',...
                '>>退出<<');                

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %JAFFE自测试
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
    if (choice ==1)
        %//////////////////////////////////////////////////////////////////
        %交叉验证，两种方式构建训练集合和测试集合
        fprintf('两种方式构建训练集和测试集:\n');
        fprintf('1：依照个体分类，9个个体数据作为训练集，1个个体数据作为测试集\n');
        fprintf('2：依照表情分类，所有个体2/3的各类表情作为训练集，1/3的各类表情作为测试集\n');
        askStr=input('请选择构建方式:','s');
        if askStr=='1'
            noStr=input('请输入作为测试的个体序号(1,2,...,10),其他个体样本作为测试 :','s');
            %1对应1:23  2对应24:45  3对应46:67 4对应68:87 5对应88:108
            %6对应1:129:,7对应130:149,8对应150:170,9对应171:191,10对应192:213
            switch str2num(noStr)  
                case 1
                    startNo=1;
                    endNo=23;
                case 2
                    startNo=24;
                    endNo=45;
                case 3
                    startNo=46;
                    endNo=67;
                case 4
                    startNo=68;
                    endNo=87;
                case 5
                    startNo=88;
                    endNo=108;
                case 6
                    startNo=1;
                    endNo=129;
                case 7
                    startNo=130;
                    endNo=149;
                case 8
                    startNo=150;
                    endNo=170;
                case 9
                    startNo=171;
                    endNo=191;
                case 10
                    startNo=192;
                    endNo=213;
                otherwise
                    msgbox('请输入1-10之间的整数！默认以第10个个体数据作为测试数据！');
                    startNo=192;
                    endNo=213;
            end
            %提取测试数据
            testP=P(:,startNo:endNo);
            testT=T(:,startNo:endNo);%标签数据
            %提取训练数据
            trainP=P;
            trainP(:,startNo:endNo)=[];%好吧，原来是这样子的，将待测试部分去掉，剩余的作为训练集
            trainT=T;
            trainT(:,startNo:endNo)=[];
            
        elseif askStr=='2'
            msgbox('还没有做！'); %666666，行吧，这里实际上就是1的内容，稍微修改一下
%             for i=1:213
%                 if samples_Data{i,}
%             end
        else
            msgbox('选择错误，请重新选择！');
            continue;
        end

       
        %//////////////////////////////////////////////////////////////////
        %开始自测试
        fprintf('<开始> JAFFE数据集自测试...\n\n');
        testP=tramnmx(testP,minP,maxP);%归一化数据集
        result=sim(Net,testP);%仿真
        result=postmnmx(result,minT,maxT);%归一化结果
        save mat_result result;%保存结果
        fprintf('<结束> JAFFE数据集自测试\n\n');
        
        %统计自测试的正确率
        [m n]=size(testP);% m为354 n为22
        count=0;%统计正确分类样本数
        num=0;%统计错误分类样本数
        for i=1:n
            [a b]=sort(result(:,i),'descend');
            if T(b(1,1),i)==1
                count=count+1;
            else
                num=num+1;
                i;
            end
        end
        str='<结束> 自测试，测试正确率 ： ';
        str=strcat(str,num2str(100*count/n));
        str=strcat(str,'%');
        disp(str);
    end    
    
    
    
                
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %训练神经网络
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (choice == 2)
        %//////////////////////////////////////////////////////////////
        %用户选项
        isTrain=0;
        if exist('mat_net.mat','file')
            askStr=input('<提示> 已有BP神经网络分类器，是否继续训练新的分类器？(Y/N) :','s');
            if askStr=='Y'||askStr=='y'
                isTrain=1;
            else
                isTrain=0;
            end   
        else
            isTrain=1;
        end
        
        %//////////////////////////////////////////////////////////////
        %根据是否存在分类器和用户选项决定是否训练新的分类器
        if isTrain==1
            %训练新的神经网络分类器
            fprintf ('<开始> 训练神经网络分类器...\n\n');
            fprintf('以全部10个个体数据作为训练数据集，并以全部数据为测试集\n');
            %训练神经网络
            [Net,minP,maxP,minT,maxT]=net_Train(trainP,trainT);
            fprintf ('<结束> 训练神经网络分类器\n\n');
        else
            fprintf ('<加载> 已有神经网络分类器\n\n');
            load('mat_net.mat');
            [PP,minP,maxP,TT,minT,maxT]=premnmx(P,T);%获得最大值、最小值
        end
        trainP=P;
        trainT=T;
        testP=P;
        testT=T;
        
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %测试图片
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (choice == 3)
        close all;
        %判断分类器是否已经存在
        if exist('Net','var')==0||exist('minP','var')==0||exist('maxP','var')==0
            msgbox('没有找到神经网络分类器，请先加载或训练！');
            continue;
        end
        
        %打开图片对话框
        [file_name file_path] = uigetfile ('*.jpg');
        if file_path ~= 0
            %读入图片
            str='<提示> 读取图片 <';
            str=strcat(str,file_name);
            str=strcat(str,'> .....\n\n');
            fprintf(str);
            I=imread([file_path,file_name]);
            figure(1),subplot(121),imshow(I),title('测试图片原图：');
            
            %旋转图片
            isGo=1;
            while(isGo==1)
                fprintf('\n');
                askStr=input('要旋转图像吗？(Y/N) :','s');
                if askStr=='Y'||askStr=='y'
                    fprintf('\n');
                    angleStr=input('请输入旋转角度(0-180°,正数对应逆时针) :','s');
                    %旋转
                    I=imrotate(I,str2num(angleStr));
                    subplot(122),imshow(I),title('旋转后图片：');
                    isGo=1;
                else
                    isGo=0;
                end
            end
            
            %切割ROI，切割好的ROI图片保存到Temp目录下
            fprintf('\n');
            askStr=input('要进行ROI提取吗?(Y/N) :','s');
            if askStr=='Y'||askStr=='y'
                isComplete=0;
                while(isComplete==0)%循环提取过程，知道提取完毕，为的是避免提取过程出错，可以再重来
                    fprintf('<提示> 正在进行ROI分割.....\n\n');
                    isComplete=getROI('mat',file_path,'Temp/',[],I,file_name);
                end
            end

            %创建测试样本集,形成一个636*1的数据矩阵testSet，也即testP
            fprintf('\n<创建> 测试测试样本集合...\n\n');
            testSetP=create_testSet(G,'Temp/',file_name);
            
            %测试样本
            fprintf('\n<开始> 分类输入样本...\n\n');
            testSetP=tramnmx(testSetP,minP,maxP);%测试数据归一化到训练数据范围
            testResult=sim(Net,testSetP);%利用已有分类器进行分类
            testResult=postmnmx(testResult,minT,maxT);%结果归一化到标签范围
%             save mat_testResult testResult;%保存结果

            %输出结果
            label=cell(1,7);
            label{1,1}='生气';            label{1,2}='恶心';            label{1,3}='恐惧';
            label{1,4}='高兴';            label{1,5}='中性';            label{1,6}='悲伤';            label{1,7}='惊讶';
            [a b]=sort(testResult(:,1),'descend');
            a=a*100;%归一化到100分
            
            str='<结果> 分类结果为： \n';
            fprintf(str);
            %第一结果
            str='<1: ';
            str=strcat(str,num2str(a(1,1)));
            str=strcat(str,'->');
            fprintf(str);
            str=label{1,b(1,1)};
            str=strcat(str,'>\n\n');
            fprintf(str);
            %第二结果
            str='<2: ';
            str=strcat(str,num2str(a(2,1)));
            str=strcat(str,'->');
            str=strcat(str,label{1,b(2,1)});
            str=strcat(str,'>\n\n');
            fprintf(str);
            %第三结果
            str='<3: ';
            str=strcat(str,num2str(a(3,1)));
            str=strcat(str,'->');
            str=strcat(str,label{1,b(3,1)});
            str=strcat(str,'>\n\n');
            fprintf(str);
            %第四结果
            str='<4: ';
            str=strcat(str,num2str(a(4,1)));
            str=strcat(str,'->');
            str=strcat(str,label{1,b(4,1)});
            str=strcat(str,'>\n\n');
            fprintf(str);
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %人脸剪切
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
    if (choice ==4)
        msgbox('已经剪切好，暂不可用');
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %人脸ROI剪切
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    if (choice == 5)
        msgbox('已经剪切好，暂不可用');
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %KNN聚类
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    if (choice == 6)
        msgbox('已经剪切好，暂不可用');
    end    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %退出
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (choice == 7)
        clear all;
        clc;
        close all;
        return;
    end    
    
end